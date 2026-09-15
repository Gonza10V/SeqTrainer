#!/usr/bin/env python3
"""Unattended Google Cloud handoff and continuation for Stage C C19.

The public commands are ``export``, ``preflight``, ``provision``, ``run``,
``status``, and ``repatriate``.  The implementation intentionally shells out
to the authenticated ``gcloud`` CLI: no long-lived service-account key is
created, and Cloud Storage's end-to-end checksum validation remains enabled.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


TRAINING_COMMIT = "ae72fae21ff9a0b50e4fe1d9d642c38c643b4923"
RUN_NAME = "c19_v3_medium_adaptive_e25"
RUN_ID = "medium_adaptive_e25_v1"
DEFAULT_PREFIX = "stage-c-c19"
DEFAULT_MACHINE = "a2-highgpu-1g"
DEFAULT_DISK_GB = 200
MAX_RUNTIME_HOURS = 36
WORKING_BUDGET_USD = 250.0
DEFAULT_HOURLY_ESTIMATE_USD = 4.25
SYSTEMD_UNIT = """[Unit]
Description=Stage C C19 continuation
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=root
WorkingDirectory=/mnt/disks/stage-c-c19
ExecStart=/usr/bin/python3 /usr/local/sbin/stage_c_c19_gcp.py run --config /mnt/disks/stage-c-c19/C19_GCP_CONFIG.json
Restart=no
TimeoutStartSec=infinity

[Install]
WantedBy=multi-user.target
"""

DEEP_FLAGS = [
    "--memory-architecture", "paper_residual_mlp_v2",
    "--memory-depth", "2",
    "--memory-expansion-factor", "4",
    "--memory-projection-convolution-kernel", "4",
    "--memory-normalize-queries-and-keys",
    "--memory-gate-granularity", "per_layer_channel",
    "--memory-recurrence-policy", "paper_exact",
    "--memory-surprise-clip-norm", "none",
    "--memory-alpha-initial", "0.001",
    "--memory-eta-initial", "0.9",
    "--memory-theta-initial", "0.001",
    "--memory-associative-loss-reduction", "sum",
    "--memory-max-gradient-rms", "none",
    "--memory-max-gradient-rms-ratio", "none",
    "--memory-theta-max", "1.0",
]


class ContractError(RuntimeError):
    """The cutover or continuation contract is not satisfied."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        chunk = handle.read(chunk_size)
        while chunk:
            digest.update(chunk)
            chunk = handle.read(chunk_size)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def run_command(
    command: Sequence[str], *, capture: bool = False, check: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), check=check, cwd=cwd, text=True,
        capture_output=capture,
    )


def json_command(command: Sequence[str]) -> Any:
    result = run_command(command, capture=True)
    return json.loads(result.stdout)


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Read owned C19 checkpoint metadata on CPU without constructing a model."""
    try:
        import torch
    except ImportError as error:
        raise ContractError("PyTorch is required to inspect latest.pt") from error
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ContractError("latest.pt is not a Stage C checkpoint mapping")
    trainer = payload.get("trainer_state")
    scheduler = payload.get("scheduler_state")
    if not isinstance(trainer, Mapping) or not isinstance(scheduler, Mapping):
        raise ContractError("latest.pt lacks trainer or scheduler state; refusing fresh C19")
    step = int(trainer.get("optimizer_step", -1))
    bases = int(trainer.get("processed_bases", -1))
    if step < 0 or bases < 0:
        raise ContractError("latest.pt has invalid optimizer step or processed bases")
    return {
        "optimizer_step": step,
        "processed_bases": bases,
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
        "dataset_components": payload.get("dataset_components"),
        "code_commit": payload.get("code_commit"),
        "model_config": payload.get("model_config"),
        "scheduler_policy": scheduler.get("policy"),
        "scheduler_batch_size": scheduler.get("batch_size"),
        "scheduler_burst_segments": scheduler.get("burst_segments"),
        "scheduler_seed": scheduler.get("seed"),
        "scheduler_state_sha256": hashlib.sha256(
            repr(sorted(scheduler.keys())).encode("utf-8")
        ).hexdigest(),
    }


def runtime_manifest() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("torch", "numpy", "pyarrow", "pandas", "transformers"):
        with suppress(Exception):
            module = __import__(name)
            packages[name] = str(module.__version__)
    cuda = None
    with suppress(Exception):
        import torch
        cuda = torch.version.cuda
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": packages,
        "cuda": cuda,
    }


def selected_export_paths(drive_root: Path) -> list[tuple[Path, Path]]:
    """Return the deliberately narrow C19 cutover allow-list."""
    study = drive_root / "study/stage_c_ecoli_medium_deep_memory_v3"
    return [
        (drive_root / "stage_c_dataset/ordered_streams/nonoverlap_6mer_v1", Path("dataset")),
        (study / "panels/e25.json", Path("panels/e25.json")),
        (study / "panels/validation.json", Path("panels/validation.json")),
        (study / "panels/panel_summary.json", Path("panels/panel_summary.json")),
        (drive_root / "runs/c18_v3_medium_a100_qualification/qualification_selection.json", Path("qualification/qualification_selection.json")),
        (study / "protocol.json", Path("study/protocol.json")),
        (study / "amendments", Path("study/amendments")),
        (study / "study_manifest.json", Path("study/study_manifest.json")),
        (study / "ledger.jsonl", Path("study/ledger.jsonl")),
        (study / "record_markers", Path("study/record_markers")),
        (drive_root / f"runs/{RUN_NAME}", Path(f"runs/{RUN_NAME}")),
    ]


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from (item for item in sorted(path.rglob("*")) if item.is_file())


def stable_copy(source: Path, destination: Path) -> dict[str, Any]:
    before = sha256_file(source)
    source_stat = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, partial)
    copied = sha256_file(partial)
    after = sha256_file(source)
    after_stat = source.stat()
    if before != copied or before != after or (
        source_stat.st_size, source_stat.st_mtime_ns
    ) != (after_stat.st_size, after_stat.st_mtime_ns):
        partial.unlink(missing_ok=True)
        raise ContractError(f"source changed during cutover: {source}")
    os.replace(partial, destination)
    return {
        "sha256": before,
        "source_sha256_before": before,
        "source_sha256_after": after,
        "copied_sha256": copied,
        "bytes": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
    }


def export_cutover(drive_root: Path, output: Path) -> dict[str, Any]:
    run_dir = drive_root / "runs" / RUN_NAME
    checkpoint = run_dir / "latest.pt"
    qualification = drive_root / "runs/c18_v3_medium_a100_qualification/qualification_selection.json"
    if not checkpoint.is_file():
        raise ContractError("C19 latest.pt is required; this workflow never starts from scratch")
    if not qualification.is_file() or not json.loads(qualification.read_text())["passed"]:
        raise ContractError("the C18 A100 qualification is missing or did not pass")
    live = run_dir / "LIVE_STATUS.json"
    if live.exists() and json.loads(live.read_text()).get("state") in {"starting", "running"}:
        raise ContractError("03l reports active training; stop the Colab process before export")
    lock = run_dir / "GCP_CUTOVER_LOCK.json"
    if lock.exists():
        raise ContractError(f"a cutover lock already exists: {lock}")
    metadata = checkpoint_metadata(checkpoint)
    cutover_id = f"step-{metadata['optimizer_step']:09d}-{sha256_file(checkpoint)[:16]}"
    bundle = output / cutover_id
    if bundle.exists():
        raise ContractError(f"immutable cutover already exists: {bundle}")
    atomic_json(lock, {
        "format_version": 1, "created_at": utc_now(), "cutover_id": cutover_id,
        "optimizer_step": metadata["optimizer_step"],
        "warning": "Do not run Colab notebook 03l while this lock exists.",
    })
    files: dict[str, Any] = {}
    try:
        for source_root, relative_root in selected_export_paths(drive_root):
            if not source_root.exists():
                if source_root.name in {"panel_summary.json", "study_manifest.json", "amendments", "record_markers"}:
                    continue
                raise ContractError(f"required cutover input is missing: {source_root}")
            for source in iter_files(source_root):
                suffix = Path(source.name) if source_root.is_file() else source.relative_to(source_root)
                relative = relative_root if source_root.is_file() else relative_root / suffix
                files[relative.as_posix()] = stable_copy(source, bundle / relative)
        checkpoint_relative = f"runs/{RUN_NAME}/latest.pt"
        if checkpoint_relative not in files:
            raise ContractError("latest.pt was not included in the cutover bundle")
        checkpoint_mtime = checkpoint.stat().st_mtime_ns
        telemetry = [
            relative for relative, record in files.items()
            if relative.startswith(f"runs/{RUN_NAME}/")
            and relative != checkpoint_relative
            and int(record["mtime_ns"]) > checkpoint_mtime
        ]
        manifest: dict[str, Any] = {
            "format_version": 1,
            "immutable": True,
            "created_at": utc_now(),
            "cutover_id": cutover_id,
            "training_commit": TRAINING_COMMIT,
            "run_name": RUN_NAME,
            "run_id": RUN_ID,
            "checkpoint": {
                **metadata,
                "relative_path": checkpoint_relative,
                "sha256": files[checkpoint_relative]["sha256"],
            },
            "runtime": runtime_manifest(),
            "files": files,
            "post_checkpoint_telemetry": telemetry,
            "post_checkpoint_telemetry_resumable": False,
        }
        atomic_json(bundle / "CUTOVER_MANIFEST.json", manifest)
        stable_copy(lock, bundle / f"runs/{RUN_NAME}/GCP_CUTOVER_LOCK.json")
        return manifest
    except BaseException:
        shutil.rmtree(bundle, ignore_errors=True)
        raise


def verify_bundle(bundle: Path, *, verify_all: bool = True) -> dict[str, Any]:
    path = bundle / "CUTOVER_MANIFEST.json"
    if not path.is_file():
        raise ContractError("CUTOVER_MANIFEST.json is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("training_commit") != TRAINING_COMMIT or manifest.get("run_id") != RUN_ID:
        raise ContractError("wrong training commit or C19 run identity")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ContractError("cutover file inventory is invalid")
    required = {
        "dataset/token_stream_manifest.json", "panels/e25.json", "panels/validation.json",
        "qualification/qualification_selection.json", "study/protocol.json", "study/ledger.jsonl",
        f"runs/{RUN_NAME}/latest.pt",
    }
    missing = required.difference(files)
    if missing:
        raise ContractError(f"cutover is missing required files: {sorted(missing)}")
    targets = files.items() if verify_all else ((name, files[name]) for name in required)
    for relative, record in targets:
        target = bundle / relative
        if not target.is_file() or sha256_file(target) != record["sha256"]:
            raise ContractError(f"cutover checksum mismatch: {relative}")
    dataset_manifest_path = bundle / "dataset/token_stream_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    index_path = bundle / "dataset" / str(dataset_manifest.get("index", ""))
    if not index_path.is_file() or sha256_file(index_path) != dataset_manifest.get("index_sha256"):
        raise ContractError("ordered dataset index checksum mismatch")
    for shard in dataset_manifest.get("shards", []):
        for field, digest_field in (("tokens", "tokens_sha256"), ("base_lengths", "base_lengths_sha256")):
            shard_path = bundle / "dataset" / str(shard.get(field, ""))
            if not shard_path.is_file() or sha256_file(shard_path) != shard.get(digest_field):
                raise ContractError(f"ordered dataset shard checksum mismatch: {shard_path.name}")
    dataset_sha = sha256_file(dataset_manifest_path)
    panel_hashes: dict[str, str] = {}
    for name, role, split in (("e25", "train", "train"), ("validation", "validation", "val")):
        panel = json.loads((bundle / f"panels/{name}.json").read_text())
        if panel.get("role") != role or panel.get("split") != split:
            raise ContractError(f"wrong {name} panel role or split")
        if panel.get("parent_dataset_fingerprint") != dataset_sha:
            raise ContractError(f"{name} panel belongs to a different ordered dataset")
        canonical = json.dumps(panel, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        panel_hashes[name] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    expected_fingerprint = hashlib.sha256(json.dumps(
        {"dataset": dataset_sha, "train_panel": panel_hashes["e25"], "validation_panel": panel_hashes["validation"]},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    checkpoint = bundle / manifest["checkpoint"]["relative_path"]
    actual = checkpoint_metadata(checkpoint)
    for key in ("optimizer_step", "processed_bases", "dataset_fingerprint"):
        if actual[key] != manifest["checkpoint"].get(key):
            raise ContractError(f"checkpoint identity mismatch for {key}")
    if actual.get("code_commit") != TRAINING_COMMIT:
        raise ContractError("checkpoint was not produced by the frozen C19 training commit")
    if actual["dataset_fingerprint"] != expected_fingerprint:
        raise ContractError("checkpoint, ordered dataset, and panels do not share one fingerprint")
    qualification = json.loads((bundle / "qualification/qualification_selection.json").read_text())
    if qualification.get("passed") is not True:
        raise ContractError("C18 qualification did not pass")
    expected_model = {
        "block_count": 12, "d_model": 256, "num_heads": 8, "persistent_tokens": 4,
        "memory_depth": 2, "memory_architecture": "paper_residual_mlp_v2",
        "memory_expansion_factor": 4, "memory_projection_convolution_kernel": 4,
        "memory_normalize_queries_and_keys": True,
        "memory_gate_granularity": "per_layer_channel",
        "memory_recurrence_policy": "paper_exact", "memory_surprise_clip_norm": None,
        "memory_associative_loss_reduction": "sum", "memory_max_gradient_rms": None,
        "memory_max_gradient_rms_ratio": None, "memory_theta_max": 1.0,
        "memory_alpha_initial": 0.001, "memory_eta_initial": 0.9,
        "memory_theta_initial": 0.001, "gradient_horizon": 3, "memory_mode": "adaptive",
    }
    model = actual.get("model_config")
    if not isinstance(model, Mapping) or any(model.get(key) != value for key, value in expected_model.items()):
        raise ContractError("checkpoint model or memory configuration drifted from notebook 03l")
    backend = model.get("backend")
    if not isinstance(backend, Mapping) or backend.get("activation_dtype") != qualification.get("activation"):
        raise ContractError("checkpoint activation does not match the C18 qualification")
    scheduler_expected = {
        "scheduler_policy": "stateful_rotation", "scheduler_batch_size": qualification.get("batch_size"),
        "scheduler_burst_segments": 96, "scheduler_seed": 20260751,
    }
    if any(actual.get(key) != value for key, value in scheduler_expected.items()):
        raise ContractError("checkpoint scheduler configuration drifted from notebook 03l")
    return manifest


def training_command(bundle: Path, run_dir: Path, python_bin: str = "") -> list[str]:
    q = json.loads((bundle / "qualification/qualification_selection.json").read_text())
    if not q.get("passed") or q.get("batch_size") is None or q.get("activation") not in {"float32", "bfloat16"}:
        raise ContractError("C18 selection is invalid")
    executable = "seqtrainer-titans-stage-c-train"
    if python_bin:
        executable = str(Path(python_bin).parent / executable)
    return [
        executable,
        "--dataset-dir", str(bundle / "dataset"),
        "--panel-manifest", str(bundle / "panels/e25.json"),
        "--validation-panel-manifest", str(bundle / "panels/validation.json"),
        "--run-dir", str(run_dir),
        "--memory-mode", "adaptive", "--horizon", "3",
        "--batch-size", str(q["batch_size"]), "--seed", "20260751",
        "--require-panel-completion", "--scheduler-policy", "stateful_rotation",
        "--scheduler-burst-segments", "96", "--checkpoint-every", "250",
        "--learning-rate", "3e-5", "--min-learning-rate", "3e-6",
        "--lr-warmup-bases", "2000000", "--lr-decay-bases", "100000000",
        "--weight-decay", "0.1", "--gradient-clip-norm", "0.5",
        "--activation", str(q["activation"]), "--block-count", "12",
        "--d-model", "256", "--num-heads", "8", "--persistent-tokens", "4",
        *DEEP_FLAGS,
        "--protocol", str(bundle / "repo/studies/stage_c_ecoli_medium_deep_memory_v3/protocol.json"),
        "--run-id", RUN_ID,
    ]


def immutable_checkpoint_key(step: int, digest: str) -> str:
    if step < 0 or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ContractError("invalid checkpoint backup identity")
    return f"checkpoints/step-{step:09d}-{digest}/latest.pt"


def choose_disk_size(input_bytes: int) -> int:
    required = (input_bytes * 3 + 2 * (1024 ** 3) - 1) // (2 * (1024 ** 3))
    return max(DEFAULT_DISK_GB, int(required))


def vm_runtime_flags() -> list[str]:
    return [
        "--maintenance-policy", "TERMINATE",
        "--max-run-duration", f"{MAX_RUNTIME_HOURS}h",
        "--instance-termination-action", "STOP",
    ]


def quota_regions(project: str) -> list[str]:
    regions = json_command(["gcloud", "compute", "regions", "list", "--project", project, "--format=json"])
    eligible = []
    for region in regions:
        quota = next((q for q in region.get("quotas", []) if q.get("metric") == "NVIDIA_A100_GPUS"), None)
        if quota and float(quota.get("limit", 0)) - float(quota.get("usage", 0)) >= 1:
            eligible.append(str(region["name"]))
    return sorted(eligible, key=lambda value: (value != "us-central1", value))


def global_gpu_available(project: str) -> bool:
    info = json_command(["gcloud", "compute", "project-info", "describe", "--project", project, "--format=json"])
    quota = next((q for q in info.get("quotas", []) if q.get("metric") in {"GPUS_ALL_REGIONS", "GPUS"}), None)
    return bool(quota and float(quota.get("limit", 0)) - float(quota.get("usage", 0)) >= 1)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    billing = json_command(["gcloud", "billing", "projects", "describe", args.project, "--format=json"])
    if not billing.get("billingEnabled"):
        raise ContractError("billing is not enabled for the project")
    run_command(["gcloud", "storage", "ls", f"gs://{args.bucket}"])
    regions = quota_regions(args.project)
    if args.region != "auto":
        regions = [region for region in regions if region == args.region]
    if not regions:
        raise ContractError("no requested region has one free NVIDIA_A100_GPUS quota unit")
    if not global_gpu_available(args.project):
        raise ContractError("global GPU quota has no free unit")
    estimate = args.hourly_cost_usd * MAX_RUNTIME_HOURS + args.storage_allowance_usd
    if estimate > WORKING_BUDGET_USD:
        raise ContractError(f"36-hour compute plus storage estimate ${estimate:.2f} exceeds $250")
    report = {
        "format_version": 1, "checked_at": utc_now(), "project": args.project,
        "bucket": args.bucket, "eligible_regions": regions,
        "selected_region": regions[0], "global_gpu_quota": True,
        "machine_type": DEFAULT_MACHINE, "gpu": "NVIDIA A100 40 GB",
        "runtime_hours": MAX_RUNTIME_HOURS, "estimated_cost_usd": estimate,
        "billing_account": billing.get("billingAccountName"),
    }
    if args.bundle:
        manifest = verify_bundle(args.bundle, verify_all=not args.fast)
        input_bytes = sum(int(row["bytes"]) for row in manifest["files"].values())
        report["cutover_id"] = manifest["cutover_id"]
        report["cutover_step"] = manifest["checkpoint"]["optimizer_step"]
        report["input_bytes"] = input_bytes
        report["disk_gb"] = choose_disk_size(input_bytes)
    if args.on_vm:
        import torch
        if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0).upper():
            raise ContractError("worker VM does not expose an NVIDIA A100")
        report["torch"] = torch.__version__
        report["cuda"] = torch.version.cuda
        if args.bundle:
            free = shutil.disk_usage(args.bundle).free
            required = int(report["input_bytes"] * 0.5) + 20 * 1024 ** 3
            if free < required:
                raise ContractError("persistent disk has insufficient free space for continuation")
    return report


def upload_immutable(source: Path, uri: str) -> None:
    result = run_command(
        ["gcloud", "storage", "cp", "--if-generation-match=0", str(source), uri],
        check=False, capture=True,
    )
    if result.returncode:
        raise ContractError(f"immutable upload collision or failure at {uri}: {result.stderr.strip()}")


def upload_bundle(bundle: Path, bucket: str, prefix: str = DEFAULT_PREFIX) -> str:
    manifest = verify_bundle(bundle)
    uri = f"gs://{bucket}/{prefix}/cutovers/{manifest['cutover_id']}"
    run_command(["gcloud", "storage", "rsync", "--recursive", str(bundle), uri])
    # Re-copy the contract with a generation precondition into a publication marker.
    upload_immutable(bundle / "CUTOVER_MANIFEST.json", f"{uri}/PUBLISHED_CUTOVER_MANIFEST.json")
    return uri


def provision(args: argparse.Namespace) -> dict[str, Any]:
    if args.bundle is None:
        raise ContractError("provision requires --bundle; a verified C19 checkpoint is mandatory")
    report = preflight(args)
    region = report["selected_region"]
    zones = json_command(["gcloud", "compute", "machine-types", "list", "--project", args.project,
                          "--filter", f"name={DEFAULT_MACHINE} AND zone:({region})", "--format=json"])
    if not zones:
        raise ContractError(f"no UP zone found in {region}")
    zone = str(zones[0]["zone"].rsplit("/", 1)[-1])
    name = args.name
    service_account = f"{name}@{args.project}.iam.gserviceaccount.com"
    project_info = json_command(["gcloud", "projects", "describe", args.project, "--format=json"])
    billing_account = str(report.get("billing_account") or "").rsplit("/", 1)[-1]
    budget_name = f"{name} $250 working cap alert"
    existing_budgets = json_command([
        "gcloud", "billing", "budgets", "list", "--billing-account", billing_account,
        "--filter", f"displayName={budget_name}", "--format=json",
    ])
    if not existing_budgets:
        run_command([
            "gcloud", "billing", "budgets", "create", "--billing-account", billing_account,
            "--display-name", budget_name, "--budget-amount", "250USD",
            "--filter-projects", f"projects/{project_info['projectId']}",
            "--threshold-rule", "percent=0.5", "--threshold-rule", "percent=0.8",
            "--threshold-rule", "percent=1.0",
        ])
    commands = [
        ["gcloud", "iam", "service-accounts", "create", name, "--project", args.project,
         "--display-name", "Stage C C19 continuation"],
        ["gcloud", "projects", "add-iam-policy-binding", args.project,
         "--member", f"serviceAccount:{service_account}", "--role", "roles/storage.objectAdmin"],
        ["gcloud", "projects", "add-iam-policy-binding", args.project,
         "--member", f"serviceAccount:{service_account}", "--role", "roles/logging.logWriter"],
        ["gcloud", "projects", "add-iam-policy-binding", args.project,
         "--member", f"serviceAccount:{service_account}", "--role", "roles/monitoring.metricWriter"],
        ["gcloud", "compute", "networks", "create", name, "--project", args.project,
         "--subnet-mode", "custom"],
        ["gcloud", "compute", "networks", "subnets", "create", name, "--project", args.project,
         "--network", name, "--region", region, "--range", "10.71.0.0/24",
         "--enable-private-ip-google-access"],
        ["gcloud", "compute", "firewall-rules", "create", f"{name}-iap-ssh", "--project", args.project,
         "--network", name, "--direction", "INGRESS", "--action", "ALLOW", "--rules", "tcp:22",
         "--source-ranges", "35.235.240.0/20"],
        ["gcloud", "compute", "routers", "create", name, "--project", args.project,
         "--network", name, "--region", region],
        ["gcloud", "compute", "routers", "nats", "create", name, "--project", args.project,
         "--router", name, "--region", region, "--auto-allocate-nat-external-ips",
         "--nat-all-subnet-ip-ranges"],
        ["gcloud", "compute", "disks", "create", name, "--project", args.project,
         "--zone", zone, "--type", "pd-ssd", "--size", f"{report.get('disk_gb', DEFAULT_DISK_GB)}GB"],
    ]
    for command in commands:
        result = run_command(command, check=False, capture=True)
        if result.returncode and "already exists" not in (result.stderr + result.stdout).lower():
            raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
    config = {
        "project": args.project, "bucket": args.bucket, "prefix": args.prefix,
        "cutover_id": report.get("cutover_id"), "zone": zone, "name": name,
    }
    with tempfile.TemporaryDirectory() as temporary:
        config_path = Path(temporary) / "C19_GCP_CONFIG.json"
        atomic_json(config_path, config)
        launcher_uri = f"gs://{args.bucket}/{args.prefix}/control/{name}/stage_c_c19_gcp.py"
        config_uri = f"gs://{args.bucket}/{args.prefix}/control/{name}/C19_GCP_CONFIG.json"
        run_command(["gcloud", "storage", "cp", str(Path(__file__).resolve()), launcher_uri])
        run_command(["gcloud", "storage", "cp", str(config_path), config_uri])
    startup = (
        "#!/bin/bash\nset -euo pipefail\n"
        "DISK=/dev/disk/by-id/google-stage-c-c19\nMOUNT=/mnt/disks/stage-c-c19\n"
        "mkdir -p $MOUNT\nif ! blkid $DISK; then mkfs.ext4 -m 0 -F $DISK; fi\n"
        "mountpoint -q $MOUNT || mount -o discard,defaults $DISK $MOUNT\n"
        f"gcloud storage cp {launcher_uri} /usr/local/sbin/stage_c_c19_gcp.py\n"
        f"gcloud storage cp {config_uri} $MOUNT/C19_GCP_CONFIG.json\n"
        "chmod 755 /usr/local/sbin/stage_c_c19_gcp.py\n"
        "cat > /etc/systemd/system/stage-c-c19.service <<'C19UNIT'\n"
        f"{SYSTEMD_UNIT}"
        "C19UNIT\n"
        "systemctl daemon-reload\nsystemctl enable --now stage-c-c19.service\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
        handle.write(startup)
        startup_path = Path(handle.name)
    try:
        run_command([
            "gcloud", "compute", "instances", "create", name, "--project", args.project,
            "--zone", zone, "--machine-type", DEFAULT_MACHINE, "--network-interface",
            f"subnet={name},no-address", "--service-account", service_account,
            "--scopes", "cloud-platform", *vm_runtime_flags(),
            "--image-family", args.image_family, "--image-project", "deeplearning-platform-release",
            "--disk", f"name={name},device-name=stage-c-c19,mode=rw,boot=no,auto-delete=no",
            "--metadata-from-file", f"startup-script={startup_path}",
        ])
    finally:
        startup_path.unlink(missing_ok=True)
    report.update({"zone": zone, "instance": name, "service_account": service_account})
    return report


def ensure_repo(bundle: Path) -> Path:
    repo = bundle / "repo"
    if not repo.exists():
        run_command(["git", "clone", "https://github.com/Gonza10V/SeqTrainer.git", str(repo)])
    run_command(["git", "fetch", "origin", TRAINING_COMMIT], cwd=repo)
    run_command(["git", "checkout", "--detach", TRAINING_COMMIT], cwd=repo)
    actual = run_command(["git", "rev-parse", "HEAD"], capture=True, cwd=repo).stdout.strip()
    if actual != TRAINING_COMMIT:
        raise ContractError("training checkout did not resolve to the frozen C19 commit")
    return repo


def newest_checkpoint_step(run_dir: Path) -> int:
    checkpoint = run_dir / "latest.pt"
    return checkpoint_metadata(checkpoint)["optimizer_step"] if checkpoint.exists() else -1


def stage_cutover(config: Mapping[str, Any], work: Path) -> tuple[Path, dict[str, Any]]:
    cutover = work / "cutover"
    uri = f"gs://{config['bucket']}/{config.get('prefix', DEFAULT_PREFIX)}/cutovers/{config['cutover_id']}"
    if not (cutover / "CUTOVER_MANIFEST.json").exists():
        cutover.mkdir(parents=True, exist_ok=True)
        run_command(["gcloud", "storage", "rsync", "--recursive", uri, str(cutover)])
    manifest = verify_bundle(cutover)
    cutover_step = int(manifest["checkpoint"]["optimizer_step"])
    run_dir = work / "authoritative" / "runs" / RUN_NAME
    disk_step = newest_checkpoint_step(run_dir)
    if disk_step < 0:
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(cutover / "runs" / RUN_NAME, run_dir)
        # Preserve the Colab RUN_START in the immutable cutover, but make the
        # cloud process declare its actual resume boundary.
        (run_dir / "RUN_START.json").unlink(missing_ok=True)
    elif disk_step < cutover_step:
        raise ContractError("persistent disk checkpoint regressed below cutover; refusing overwrite")
    return cutover, manifest


def mirror_outputs(run_dir: Path, config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = run_dir / "latest.pt"
    snapshot = run_dir / ".gcp_checkpoint_snapshot.pt"
    for _ in range(3):
        before = checkpoint.stat()
        partial = snapshot.with_suffix(".pt.partial")
        shutil.copy2(checkpoint, partial)
        after = checkpoint.stat()
        identity_before = (before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before == identity_after:
            os.replace(partial, snapshot)
            break
        partial.unlink(missing_ok=True)
    else:
        raise ContractError("checkpoint changed repeatedly while preparing its immutable backup")
    snapshot_metadata = checkpoint_metadata(snapshot)
    step = int(snapshot_metadata["optimizer_step"])
    digest = sha256_file(snapshot)
    root = f"gs://{config['bucket']}/{config.get('prefix', DEFAULT_PREFIX)}/runs/{RUN_NAME}"
    key = immutable_checkpoint_key(step, digest)
    marker = run_dir / ".last_uploaded_checkpoint"
    if marker.exists():
        previous_key = marker.read_text().strip()
        try:
            previous_step = int(previous_key.split("step-", 1)[1].split("-", 1)[0])
        except (IndexError, ValueError) as error:
            raise ContractError("invalid persistent checkpoint backup marker") from error
        if step < previous_step:
            raise ContractError(
                f"unexplained checkpoint regression from step {previous_step} to {step}"
            )
    if not marker.exists() or marker.read_text().strip() != key:
        upload_immutable(snapshot, f"{root}/{key}")
        marker.write_text(key + "\n", encoding="utf-8")
    snapshot.unlink(missing_ok=True)
    for name in (
        "LIVE_STATUS.json", "training_history.json", "run_manifest.json", "RUN_START.json",
        "FAILED.txt", "GCP_FAILURE.json", "COMPLETE.json",
    ):
        path = run_dir / name
        if path.exists():
            run_command(["gcloud", "storage", "cp", str(path), f"{root}/live/{name}"])
    if (run_dir / "logs").is_dir():
        run_command(["gcloud", "storage", "rsync", "--recursive", str(run_dir / "logs"), f"{root}/live/logs"])
    status = {
        "format_version": 1, "updated_at": utc_now(), "state": "running",
        "optimizer_step": step, "processed_bases": snapshot_metadata["processed_bases"],
        "cutover_step": manifest["checkpoint"]["optimizer_step"], "newest_checkpoint_backup": key,
    }
    if (run_dir / "LIVE_STATUS.json").exists():
        status["state"] = json.loads((run_dir / "LIVE_STATUS.json").read_text()).get("state", "running")
    atomic_json(run_dir / "GCP_STATUS.json", status)
    run_command(["gcloud", "storage", "cp", str(run_dir / "GCP_STATUS.json"), f"{root}/live/GCP_STATUS.json"])
    return status


def run_worker(config_path: Path) -> int:
    config = json.loads(config_path.read_text())
    work = config_path.parent
    cutover, manifest = stage_cutover(config, work)
    run_dir = work / "authoritative/runs" / RUN_NAME
    input_bytes = sum(int(row["bytes"]) for row in manifest["files"].values())
    required_free = input_bytes // 2 + 20 * 1024 ** 3
    if shutil.disk_usage(work).free < required_free:
        raise ContractError("persistent disk has insufficient free capacity for checkpoints and outputs")
    if (run_dir / "GCP_FAILURE.json").exists():
        raise ContractError(
            "a terminal failure is preserved on disk; inspect it and remove GCP_FAILURE.json explicitly before retry"
        )
    if (run_dir / "COMPLETE.json").exists():
        validate_completion(run_dir, int(manifest["checkpoint"]["optimizer_step"]))
        mirror_outputs(run_dir, config, manifest)
        run_command(["shutdown", "-h", "now"], check=False)
        return 0
    repo = ensure_repo(cutover)
    # The Deep Learning image supplies CUDA. Recreate the cutover package versions in an isolated venv.
    venv = work / "venv"
    if not (venv / "bin/python").exists():
        requested_python = str(manifest["runtime"].get("python", ""))
        if requested_python and tuple(requested_python.split(".")[:2]) != tuple(platform.python_version().split(".")[:2]):
            raise ContractError(
                f"Deep Learning image Python {platform.python_version()} does not match cutover {requested_python}"
            )
        run_command([sys.executable, "-m", "venv", str(venv)])
        packages = manifest["runtime"].get("packages", {})
        pins = [f"{name}=={version}" for name, version in packages.items() if name != "torch"]
        if packages.get("torch"):
            torch_version = str(packages["torch"]).split("+", 1)[0]
            cuda_digits = str(manifest["runtime"].get("cuda") or "").replace(".", "")
            torch_install = [str(venv / "bin/pip"), "install", f"torch=={torch_version}"]
            if cuda_digits:
                torch_install.extend(["--index-url", f"https://download.pytorch.org/whl/cu{cuda_digits}"])
            run_command(torch_install)
        run_command([str(venv / "bin/pip"), "install", *pins, "-e", f"{repo}[torch,bacteria-titan]"])
    gpu_check = run_command([
        str(venv / "bin/python"), "-c",
        "import torch; assert torch.cuda.is_available(); assert 'A100' in torch.cuda.get_device_name(0).upper(); print(torch.__version__, torch.version.cuda)",
    ], capture=True)
    if not gpu_check.stdout.strip():
        raise ContractError("CUDA/PyTorch A100 preflight produced no result")
    command = training_command(cutover, run_dir, str(venv / "bin/python"))
    expected = int(manifest["checkpoint"]["optimizer_step"])
    verify_output = run_dir / "pre_resume_verification.json"
    run_command([
        str(venv / "bin/seqtrainer-titans-stage-c-resume-verify"),
        "--dataset-dir", str(cutover / "dataset"), "--panel-manifest", str(cutover / "panels/e25.json"),
        "--checkpoint", str(run_dir / "latest.pt"), "--output", str(verify_output),
        "--device", "cuda", "--expected-step", str(newest_checkpoint_step(run_dir)),
    ], cwd=repo)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "gcp_train.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, text=True)
        last_mtime_ns = -1
        last_mirror_at = 0.0
        try:
            while process.poll() is None:
                time.sleep(1)
                latest = run_dir / "latest.pt"
                current_mtime_ns = latest.stat().st_mtime_ns if latest.exists() else -1
                now = time.monotonic()
                if current_mtime_ns != last_mtime_ns or now - last_mirror_at >= 60:
                    mirror_outputs(run_dir, config, manifest)
                    last_mtime_ns = current_mtime_ns
                    last_mirror_at = now
        except BaseException as error:
            process.terminate()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=30)
            if process.poll() is None:
                process.kill()
            atomic_json(run_dir / "GCP_FAILURE.json", {
                "failed_at": utc_now(), "error_type": type(error).__name__, "error": str(error),
            })
            failure_root = (
                f"gs://{config['bucket']}/{config.get('prefix', DEFAULT_PREFIX)}"
                f"/runs/{RUN_NAME}/failure"
            )
            run_command(
                ["gcloud", "storage", "cp", str(run_dir / "GCP_FAILURE.json"), f"{failure_root}/GCP_FAILURE.json"],
                check=False,
            )
            run_command(
                ["gcloud", "storage", "rsync", "--recursive", str(log_dir), f"{failure_root}/logs"],
                check=False,
            )
            raise
    if process.returncode:
        atomic_json(run_dir / "GCP_FAILURE.json", {"failed_at": utc_now(), "returncode": process.returncode})
        if (run_dir / "latest.pt").exists():
            mirror_outputs(run_dir, config, manifest)
        return int(process.returncode)
    start = json.loads((run_dir / "RUN_START.json").read_text())
    if int(start["starting_optimizer_steps"]) != expected:
        raise ContractError("trainer did not report the cutover optimizer step")
    final = json.loads((run_dir / "run_manifest.json").read_text())
    status = json.loads((run_dir / "LIVE_STATUS.json").read_text())
    if not (final.get("scheduler_exhausted") is True and final.get("stop_reason") == "panel_exhausted" and status.get("state") == "completed"):
        raise ContractError("trainer exited without complete-panel success")
    run_command([str(venv / "bin/seqtrainer-titans-stage-c-architecture"), "--checkpoint", str(run_dir / "latest.pt"), "--output-dir", str(run_dir)], cwd=repo)
    run_command([str(venv / "bin/seqtrainer-titans-stage-c-resume-verify"), "--dataset-dir", str(cutover / "dataset"), "--panel-manifest", str(cutover / "panels/e25.json"), "--checkpoint", str(run_dir / "latest.pt"), "--output", str(run_dir / "resume_verification.json"), "--device", "cuda"], cwd=repo)
    run_command([str(venv / "bin/seqtrainer-titans-stage-c-study"), "record", "--protocol", str(repo / "studies/stage_c_ecoli_medium_deep_memory_v3/protocol.json"), "--study-root", str(cutover / "study"), "--run-id", RUN_ID, "--evidence-tier", "exploratory", "--artifact", str(run_dir)], cwd=repo)
    marker = cutover / "study/record_markers" / f"{RUN_ID}.json"
    atomic_json(marker, {"run_id": RUN_ID, "artifact": str(run_dir), "recorded_at": utc_now()})
    study_return = run_dir / "GCP_STUDY_RETURN"
    if study_return.exists():
        shutil.rmtree(study_return)
    shutil.copytree(cutover / "study", study_return)
    final_step = newest_checkpoint_step(run_dir)
    if final_step <= expected:
        raise ContractError("acceptance requires at least one optimizer step beyond cutover")
    complete = {
        "format_version": 1, "completed_at": utc_now(), "scheduler_exhausted": True,
        "stop_reason": "panel_exhausted", "live_status": "completed",
        "architecture_passed": (run_dir / "MODEL_ARCHITECTURE.json").exists(),
        "resume_verification_passed": json.loads((run_dir / "resume_verification.json").read_text()).get("status") == "passed",
        "cutover_step": expected, "final_step": final_step,
        "checkpoint_sha256": sha256_file(run_dir / "latest.pt"),
    }
    atomic_json(run_dir / "COMPLETE.json", complete)
    mirror_outputs(run_dir, config, manifest)
    root = f"gs://{config['bucket']}/{config.get('prefix', DEFAULT_PREFIX)}/runs/{RUN_NAME}/complete"
    run_command(["gcloud", "storage", "rsync", "--recursive", str(run_dir), root])
    run_command(["shutdown", "-h", "now"], check=False)
    return 0


def status(args: argparse.Namespace) -> dict[str, Any]:
    instance = json_command(["gcloud", "compute", "instances", "describe", args.name,
                             "--project", args.project, "--zone", args.zone, "--format=json"])
    result: dict[str, Any] = {"instance": args.name, "vm_state": instance.get("status")}
    uri = f"gs://{args.bucket}/{args.prefix}/runs/{RUN_NAME}/live/GCP_STATUS.json"
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "status.json"
        downloaded = run_command(["gcloud", "storage", "cp", uri, str(target)], check=False, capture=True)
        if downloaded.returncode == 0:
            result["training"] = json.loads(target.read_text())
    logs = run_command(["gcloud", "compute", "ssh", args.name, "--project", args.project,
                        "--zone", args.zone, "--tunnel-through-iap", "--command",
                        "sudo journalctl -u stage-c-c19.service -n 30 --no-pager"],
                       check=False, capture=True)
    result["recent_logs"] = logs.stdout[-12000:]
    return result


def validate_completion(run_dir: Path, cutover_step: int) -> dict[str, Any]:
    complete = json.loads((run_dir / "COMPLETE.json").read_text())
    final = json.loads((run_dir / "run_manifest.json").read_text())
    live = json.loads((run_dir / "LIVE_STATUS.json").read_text())
    valid = (
        final.get("scheduler_exhausted") is True
        and final.get("stop_reason") == "panel_exhausted"
        and live.get("state") == "completed"
        and complete.get("architecture_passed") is True
        and complete.get("resume_verification_passed") is True
        and int(complete.get("final_step", -1)) > cutover_step
        and sha256_file(run_dir / "latest.pt") == complete.get("checkpoint_sha256")
    )
    if not valid:
        raise ContractError("completed bundle failed the guarded return contract")
    return complete


def atomic_repatriate(downloaded_run: Path, drive_root: Path, cutover_manifest: Mapping[str, Any]) -> None:
    cutover_step = int(cutover_manifest["checkpoint"]["optimizer_step"])
    complete = validate_completion(downloaded_run, cutover_step)
    destination = drive_root / "runs" / RUN_NAME
    current = destination / "latest.pt"
    if not current.exists() or newest_checkpoint_step(destination) != cutover_step:
        raise ContractError("Drive C19 independently advanced or lost its cutover checkpoint")
    if sha256_file(current) != cutover_manifest["checkpoint"]["sha256"]:
        raise ContractError("Drive cutover checkpoint hash changed")
    study_return = downloaded_run / "GCP_STUDY_RETURN"
    if not study_return.is_dir() or not (study_return / "ledger.jsonl").is_file():
        raise ContractError("completed bundle is missing the 03l study-ledger record")
    study_root = drive_root / "study/stage_c_ecoli_medium_deep_memory_v3"
    source_ledger = cutover_manifest.get("files", {}).get("study/ledger.jsonl")
    drive_ledger = study_root / "ledger.jsonl"
    if (
        not isinstance(source_ledger, Mapping)
        or not drive_ledger.is_file()
        or sha256_file(drive_ledger) != source_ledger.get("sha256")
    ):
        raise ContractError("Drive study ledger independently advanced or changed since cutover")
    archive = destination / "pre_gcp_cutover_archive" / cutover_manifest["cutover_id"]
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(current, archive / "latest.pt")
    for source in sorted(downloaded_run.rglob("*")):
        if not source.is_file() or source.name.endswith(".partial"):
            continue
        relative = source.relative_to(downloaded_run)
        if relative.parts[0] == "GCP_STUDY_RETURN":
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        shutil.copy2(source, partial)
        if sha256_file(partial) != sha256_file(source):
            raise ContractError(f"download publication checksum failed: {relative}")
        os.replace(partial, target)
    study_files = [source for source in sorted(study_return.rglob("*")) if source.is_file()]
    # Publish the hash-chained ledger last, after every artifact it references.
    study_files.sort(key=lambda source: source.name == "ledger.jsonl")
    for source in study_files:
        relative = source.relative_to(study_return)
        target = study_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        shutil.copy2(source, partial)
        if sha256_file(partial) != sha256_file(source):
            raise ContractError(f"study publication checksum failed: {relative}")
        os.replace(partial, target)
    if newest_checkpoint_step(destination) != int(complete["final_step"]):
        raise ContractError("published Drive checkpoint step is wrong")


def repatriate(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run = root / "run"
        run.mkdir()
        uri = f"gs://{args.bucket}/{args.prefix}/runs/{RUN_NAME}/complete"
        run_command(["gcloud", "storage", "rsync", "--recursive", uri, str(run)])
        manifest_path = args.cutover_manifest
        manifest = json.loads(manifest_path.read_text())
        atomic_repatriate(run, args.drive_root, manifest)
        return {"status": "published", "final_step": newest_checkpoint_step(args.drive_root / "runs" / RUN_NAME)}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="make a stable, immutable Drive cutover bundle")
    export.add_argument("--drive-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--bucket")
    export.add_argument("--prefix", default=DEFAULT_PREFIX)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project", required=True)
    common.add_argument("--bucket", required=True)
    common.add_argument("--region", default="auto")
    common.add_argument("--bundle", type=Path)
    common.add_argument("--hourly-cost-usd", type=float, default=DEFAULT_HOURLY_ESTIMATE_USD)
    common.add_argument("--storage-allowance-usd", type=float, default=25.0)
    common.add_argument("--fast", action="store_true")
    common.add_argument("--on-vm", action="store_true")
    sub.add_parser("preflight", parents=[common], help="validate cloud, quota, data, budget, and GPU")
    provision_cmd = sub.add_parser("provision", parents=[common], help="create private A100 worker resources")
    provision_cmd.add_argument("--name", default="stage-c-c19")
    provision_cmd.add_argument("--prefix", default=DEFAULT_PREFIX)
    provision_cmd.add_argument("--image-family", default="common-cu129-ubuntu-2204-nvidia-580")
    run = sub.add_parser("run", help="service entry point; resume the authoritative disk run")
    run.add_argument("--config", type=Path, required=True)
    stat = sub.add_parser("status", help="show VM, training progress, backup, and logs")
    stat.add_argument("--project", required=True)
    stat.add_argument("--bucket", required=True)
    stat.add_argument("--zone", required=True)
    stat.add_argument("--name", default="stage-c-c19")
    stat.add_argument("--prefix", default=DEFAULT_PREFIX)
    rep = sub.add_parser("repatriate", help="atomically return a completed run to Drive")
    rep.add_argument("--bucket", required=True)
    rep.add_argument("--prefix", default=DEFAULT_PREFIX)
    rep.add_argument("--drive-root", type=Path, required=True)
    rep.add_argument("--cutover-manifest", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "export":
            manifest = export_cutover(args.drive_root, args.output)
            result: Any = manifest
            if args.bucket:
                result = {"manifest": manifest, "gcs_uri": upload_bundle(args.output / manifest["cutover_id"], args.bucket, args.prefix)}
        elif args.command == "preflight":
            result = preflight(args)
        elif args.command == "provision":
            result = provision(args)
        elif args.command == "run":
            return run_worker(args.config)
        elif args.command == "status":
            result = status(args)
        elif args.command == "repatriate":
            result = repatriate(args)
        else:
            raise AssertionError(args.command)
    except (ContractError, subprocess.CalledProcessError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
