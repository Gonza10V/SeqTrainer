#!/usr/bin/env python3
"""Gated Google Cloud runner for the matched Stage C no-memory E25 control.

This is deliberately separate from ``stage_c_c19_gcp.py``: C19 remains a
resume-only adaptive continuation, while this control is initialized exactly
once and subsequently resumes only its own no-memory checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load support module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load("stage_c_c19_gcp_support", ROOT / "stage_c_c19_gcp.py")
curc = _load("stage_c_trailblazer_support", ROOT / "stage_c_c20_curc.py")

TRAINING_COMMIT = curc.TRAINING_COMMIT
RUN_NAME = "c21_v3_medium_no_memory_e25"
RUN_ID = "medium_no_memory_e25_v1"
PILOT_RUN_ID = "medium_no_memory_e25_pilot_v1"
DEFAULT_PREFIX = "stage-c-controls/no-memory-e25"
DEFAULT_NAME = "stage-c-no-memory-e25"
DEFAULT_MACHINE = "a2-highgpu-1g"
MAX_SESSION_HOURS = 36.0
WORKING_BUDGET_USD = 250.0
DEFAULT_HOURLY_USD = 3.673385
DEFAULT_STORAGE_USD = 20.0
PILOT_STEPS = 10

SYSTEMD_UNIT = """[Unit]
Description=Stage C no-memory E25 control
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=root
WorkingDirectory=/mnt/disks/stage-c-control
ExecStart=/usr/bin/python3 /usr/local/sbin/stage_c_e25_control_gcp.py run --config /mnt/disks/stage-c-control/CONTROL_GCP_CONFIG.json
Restart=no
TimeoutStartSec=infinity

[Install]
WantedBy=multi-user.target
"""


class ContractError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def cost_projection(
    predictable_bases: int, bases_per_second: float, *, hourly_usd: float = DEFAULT_HOURLY_USD,
    storage_usd: float = DEFAULT_STORAGE_USD, prior_compute_hours: float = 0.0,
) -> dict[str, float]:
    if predictable_bases <= 0 or bases_per_second <= 0 or hourly_usd <= 0 or storage_usd < 0:
        raise ContractError("invalid inputs for the no-memory cost projection")
    training_hours = predictable_bases / bases_per_second / 3600
    compute_hours = prior_compute_hours + training_hours
    total = compute_hours * hourly_usd + storage_usd
    return {
        "training_hours": training_hours, "compute_hours": compute_hours,
        "hourly_usd": hourly_usd, "storage_usd": storage_usd,
        "estimated_total_usd": total,
        "within_working_budget": total <= WORKING_BUDGET_USD,
    }


def cumulative_runtime_cap(projected_hours: float, hourly_usd: float, storage_usd: float) -> float:
    """Add modest runtime margin without ever consuming the $50 reserve."""
    if projected_hours <= 0 or hourly_usd <= 0 or not 0 <= storage_usd < WORKING_BUDGET_USD:
        raise ContractError("invalid cumulative runtime cap inputs")
    budget_hours = (WORKING_BUDGET_USD - storage_usd) / hourly_usd
    return min(projected_hours * 1.05, budget_hours)


def training_command(
    bundle: Path, run_dir: Path, python_bin: Path, *, pilot: bool = False,
) -> list[str]:
    curc.verify_input(bundle, verify_all=False)
    q = json.loads((bundle / "qualification/qualification_selection.json").read_text())
    executable = str(python_bin.parent / "seqtrainer-titans-stage-c-train")
    command = [
        executable, "--dataset-dir", str(bundle / "dataset"),
        "--panel-manifest", str(bundle / "panels/e25.json"),
        "--validation-panel-manifest", str(bundle / "panels/validation.json"),
        "--run-dir", str(run_dir), "--memory-mode", "no_memory",
        "--horizon", "3", "--batch-size", str(q["batch_size"]), "--seed", "20260751",
        "--scheduler-policy", "stateful_rotation", "--scheduler-burst-segments", "96",
        "--checkpoint-every", "250", "--learning-rate", "3e-5",
        "--min-learning-rate", "3e-6", "--lr-warmup-bases", "2000000",
        "--lr-decay-bases", "100000000", "--weight-decay", "0.1",
        "--gradient-clip-norm", "0.5", "--activation", str(q["activation"]),
        "--block-count", "12", "--d-model", "256", "--num-heads", "8",
        "--persistent-tokens", "4", "--validation-streams", "1",
        "--validation-segments", "1",
        *curc.DEEP_FLAGS,
        "--protocol", str(bundle / "study/protocol.json"),
        "--protocol-amendment", str(bundle / f"study/{curc.AMENDMENT_NAME}"),
        "--run-id", PILOT_RUN_ID if pilot else RUN_ID,
    ]
    if pilot:
        command.extend(["--max-optimizer-steps", str(PILOT_STEPS)])
    else:
        command.append("--require-panel-completion")
    return command


def verify_control_checkpoint(path: Path) -> dict[str, Any]:
    metadata = curc.checkpoint_metadata(path)
    if metadata["code_commit"] != TRAINING_COMMIT:
        raise ContractError("control checkpoint came from the wrong training commit")
    if metadata["model_config"].get("memory_mode") != "no_memory":
        raise ContractError("persistent checkpoint is not the no-memory control")
    return metadata


def pilot_report(run_dir: Path, elapsed_seconds: float) -> dict[str, Any]:
    metadata = verify_control_checkpoint(run_dir / "latest.pt")
    if metadata["optimizer_step"] != PILOT_STEPS or metadata["processed_bases"] <= 0:
        raise ContractError("pilot did not complete the declared optimizer-step budget")
    rate = metadata["processed_bases"] / elapsed_seconds
    return {
        "format_version": 1, "completed_at": utc_now(), "optimizer_steps": PILOT_STEPS,
        "processed_bases": metadata["processed_bases"], "elapsed_seconds": elapsed_seconds,
        "bases_per_second": rate,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    manifest = curc.verify_input(args.bundle, verify_all=not args.fast)
    billing = base.json_command(["gcloud", "billing", "projects", "describe", args.project, "--format=json"])
    if not billing.get("billingEnabled"):
        raise ContractError("billing is not enabled")
    base.run_command(["gcloud", "storage", "ls", f"gs://{args.bucket}"])
    regions = base.quota_regions(args.project)
    if args.region != "auto":
        regions = [item for item in regions if item == args.region]
    if not regions or not base.global_gpu_available(args.project):
        raise ContractError("regional NVIDIA_A100_GPUS and global GPU quota are both required")
    e25 = json.loads((args.bundle / "panels/e25.json").read_text())
    report: dict[str, Any] = {
        "format_version": 1, "checked_at": utc_now(), "project": args.project,
        "bucket": args.bucket, "selected_region": regions[0], "eligible_regions": regions,
        "machine_type": DEFAULT_MACHINE, "gpu": "NVIDIA A100 40 GB",
        "bundle_id": manifest["bundle_id"], "phase": args.phase,
        "billing_account": billing.get("billingAccountName"),
    }
    if args.phase == "pilot":
        estimate = args.pilot_hours * args.hourly_cost_usd + args.storage_allowance_usd
        if estimate > WORKING_BUDGET_USD:
            raise ContractError("pilot allowance exceeds the $250 working budget")
        report["estimated_total_usd"] = estimate
        report["max_total_hours"] = args.pilot_hours
    else:
        if not args.pilot_report or not args.pilot_report.is_file():
            raise ContractError("production preflight requires a downloaded pilot report")
        pilot = json.loads(args.pilot_report.read_text())
        projection = cost_projection(
            int(e25["predictable_bases"]), float(pilot["bases_per_second"]),
            hourly_usd=args.hourly_cost_usd, storage_usd=args.storage_allowance_usd,
            prior_compute_hours=float(pilot.get("elapsed_seconds", 0)) / 3600,
        )
        if not projection["within_working_budget"]:
            raise ContractError(
                f"projected standard-A100 total ${projection['estimated_total_usd']:.2f} exceeds $250; defer the control to CURC"
            )
        report["projection"] = projection
        report["max_total_hours"] = cumulative_runtime_cap(
            projection["compute_hours"], args.hourly_cost_usd, args.storage_allowance_usd,
        )
    input_bytes = sum(int(row["bytes"]) for row in manifest["files"].values())
    report["input_bytes"] = input_bytes
    report["disk_gb"] = base.choose_disk_size(input_bytes)
    return report


def _upload_bundle(bundle: Path, bucket: str, prefix: str) -> str:
    manifest = curc.verify_input(bundle)
    uri = f"gs://{bucket}/{prefix}/inputs/{manifest['bundle_id']}"
    marker = f"{uri}/PUBLISHED_TRAILBLAZER_INPUT_MANIFEST.json"
    existing = base.run_command(["gcloud", "storage", "ls", marker], capture=True, check=False)
    if existing.returncode == 0:
        # Reuse is allowed only when the immutable publication marker has the
        # exact local identity; download and compare before touching the prefix.
        with tempfile.TemporaryDirectory() as temporary:
            remote = Path(temporary) / "manifest.json"
            base.run_command(["gcloud", "storage", "cp", marker, str(remote)])
            if curc.sha256_file(remote) != curc.sha256_file(bundle / "TRAILBLAZER_INPUT_MANIFEST.json"):
                raise ContractError(f"GCS input prefix already contains a different immutable bundle: {uri}")
        return uri
    base.run_command(["gcloud", "storage", "rsync", "--recursive", str(bundle), uri])
    base.upload_immutable(bundle / "TRAILBLAZER_INPUT_MANIFEST.json", marker)
    return uri


def provision(args: argparse.Namespace) -> dict[str, Any]:
    report = preflight(args)
    region = report["selected_region"]
    zones = base.json_command([
        "gcloud", "compute", "machine-types", "list", "--project", args.project,
        "--filter", f"name={DEFAULT_MACHINE} AND zone:({region})", "--format=json",
    ])
    if not zones:
        raise ContractError(f"no {DEFAULT_MACHINE} zone is visible in {region}")
    zone = str(zones[0]["zone"]).rsplit("/", 1)[-1]
    name = args.name
    service_account = f"{name}@{args.project}.iam.gserviceaccount.com"
    input_uri = _upload_bundle(args.bundle, args.bucket, args.prefix)
    billing_account = str(report.get("billing_account") or "").rsplit("/", 1)[-1]
    project_info = base.json_command(["gcloud", "projects", "describe", args.project, "--format=json"])
    budget_name = f"{name} $250 working cap alert"
    existing_budgets = base.json_command([
        "gcloud", "billing", "budgets", "list", "--billing-account", billing_account,
        "--filter", f"displayName={budget_name}", "--format=json",
    ])
    if not existing_budgets:
        base.run_command([
            "gcloud", "billing", "budgets", "create", "--billing-account", billing_account,
            "--display-name", budget_name, "--budget-amount", "250USD",
            "--filter-projects", f"projects/{project_info['projectId']}",
            "--threshold-rule", "percent=0.5", "--threshold-rule", "percent=0.8",
            "--threshold-rule", "percent=1.0",
        ])
    config = {
        "project": args.project, "bucket": args.bucket, "prefix": args.prefix,
        "name": name, "zone": zone, "input_uri": input_uri,
        "phase": args.phase, "max_total_hours": report["max_total_hours"],
        "bundle_id": report["bundle_id"],
    }
    commands = [
        ["gcloud", "iam", "service-accounts", "create", name, "--project", args.project,
         "--display-name", "Stage C no-memory E25 control"],
        ["gcloud", "projects", "add-iam-policy-binding", args.project,
         "--member", f"serviceAccount:{service_account}", "--role", "roles/storage.objectAdmin"],
        ["gcloud", "projects", "add-iam-policy-binding", args.project,
         "--member", f"serviceAccount:{service_account}", "--role", "roles/logging.logWriter"],
        ["gcloud", "compute", "networks", "create", name, "--project", args.project, "--subnet-mode", "custom"],
        ["gcloud", "compute", "networks", "subnets", "create", name, "--project", args.project,
         "--network", name, "--region", region, "--range", "10.72.0.0/24", "--enable-private-ip-google-access"],
        ["gcloud", "compute", "firewall-rules", "create", f"{name}-iap-ssh", "--project", args.project,
         "--network", name, "--direction", "INGRESS", "--action", "ALLOW", "--rules", "tcp:22",
         "--source-ranges", "35.235.240.0/20"],
        ["gcloud", "compute", "routers", "create", name, "--project", args.project, "--network", name, "--region", region],
        ["gcloud", "compute", "routers", "nats", "create", name, "--project", args.project,
         "--router", name, "--region", region, "--auto-allocate-nat-external-ips", "--nat-all-subnet-ip-ranges"],
        ["gcloud", "compute", "disks", "create", name, "--project", args.project, "--zone", zone,
         "--type", "pd-ssd", "--size", f"{report['disk_gb']}GB"],
    ]
    for command in commands:
        result = base.run_command(command, capture=True, check=False)
        if result.returncode and "already exists" not in (result.stdout + result.stderr).lower():
            raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        config_path = temp / "CONTROL_GCP_CONFIG.json"
        atomic_json(config_path, config)
        script_uri = f"gs://{args.bucket}/{args.prefix}/control/stage_c_e25_control_gcp.py"
        curc_uri = f"gs://{args.bucket}/{args.prefix}/control/stage_c_c20_curc.py"
        base_uri = f"gs://{args.bucket}/{args.prefix}/control/stage_c_c19_gcp.py"
        config_uri = f"gs://{args.bucket}/{args.prefix}/control/CONTROL_GCP_CONFIG.json"
        for source, uri in ((Path(__file__), script_uri), (ROOT / "stage_c_c20_curc.py", curc_uri),
                            (ROOT / "stage_c_c19_gcp.py", base_uri), (config_path, config_uri)):
            base.run_command(["gcloud", "storage", "cp", str(source), uri])
        startup = f"""#!/bin/bash
set -euo pipefail
DISK=/dev/disk/by-id/google-stage-c-control
MOUNT=/mnt/disks/stage-c-control
mkdir -p $MOUNT
if ! blkid $DISK; then mkfs.ext4 -m 0 -F $DISK; fi
mountpoint -q $MOUNT || mount -o discard,defaults $DISK $MOUNT
gcloud storage cp {script_uri} /usr/local/sbin/stage_c_e25_control_gcp.py
gcloud storage cp {curc_uri} /usr/local/sbin/stage_c_c20_curc.py
gcloud storage cp {base_uri} /usr/local/sbin/stage_c_c19_gcp.py
gcloud storage cp {config_uri} $MOUNT/CONTROL_GCP_CONFIG.json
chmod 755 /usr/local/sbin/stage_c_*_gcp.py /usr/local/sbin/stage_c_c20_curc.py
cat > /etc/systemd/system/stage-c-control.service <<'CONTROLUNIT'
{SYSTEMD_UNIT}CONTROLUNIT
systemctl daemon-reload
systemctl enable --now stage-c-control.service
"""
        startup_path = temp / "startup.sh"
        startup_path.write_text(startup, encoding="utf-8")
        create = base.run_command([
            "gcloud", "compute", "instances", "create", name, "--project", args.project,
            "--zone", zone, "--machine-type", DEFAULT_MACHINE,
            "--network-interface", f"subnet={name},no-address", "--service-account", service_account,
            "--scopes", "cloud-platform", "--maintenance-policy", "TERMINATE",
            "--max-run-duration", "36h", "--instance-termination-action", "STOP",
            "--image-family", args.image_family, "--image-project", "deeplearning-platform-release",
            "--disk", f"name={name},device-name=stage-c-control,mode=rw,boot=no,auto-delete=no",
            "--metadata-from-file", f"startup-script={startup_path}",
        ], capture=True, check=False)
        if create.returncode and "already exists" not in (create.stdout + create.stderr).lower():
            raise subprocess.CalledProcessError(create.returncode, create.args, create.stdout, create.stderr)
        if create.returncode:
            base.run_command(["gcloud", "compute", "instances", "start", name, "--project", args.project, "--zone", zone])
    report.update({"instance": name, "zone": zone, "input_uri": input_uri})
    return report


def _backup_checkpoint(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any] | None:
    source = run_dir / "latest.pt"
    if not source.is_file():
        return None
    metadata = verify_control_checkpoint(source)
    digest = curc.sha256_file(source)
    marker = run_dir / ".last_uploaded_checkpoint.json"
    if marker.is_file():
        previous = json.loads(marker.read_text())
        if int(previous["optimizer_step"]) > metadata["optimizer_step"]:
            raise ContractError("no-memory checkpoint regressed")
        if previous.get("sha256") == digest:
            return previous
    snapshot = run_dir / ".checkpoint_upload.pt"
    curc.stable_copy(source, snapshot)
    uri = (
        f"gs://{config['bucket']}/{config['prefix']}/checkpoints/"
        f"step-{metadata['optimizer_step']:09d}-{digest}/latest.pt"
    )
    base.upload_immutable(snapshot, uri)
    snapshot.unlink(missing_ok=True)
    record = {"optimizer_step": metadata["optimizer_step"], "processed_bases": metadata["processed_bases"],
              "sha256": digest, "uri": uri, "uploaded_at": utc_now()}
    atomic_json(marker, record)
    return record


def _ensure_runtime(work: Path, manifest: Mapping[str, Any]) -> tuple[Path, Path]:
    repo = work / "training-source"
    if not repo.exists():
        base.run_command(["git", "clone", "https://github.com/Gonza10V/SeqTrainer.git", str(repo)])
    base.run_command(["git", "-C", str(repo), "fetch", "origin", TRAINING_COMMIT])
    base.run_command(["git", "-C", str(repo), "checkout", "--detach", TRAINING_COMMIT])
    if base.run_command(["git", "-C", str(repo), "rev-parse", "HEAD"], capture=True).stdout.strip() != TRAINING_COMMIT:
        raise ContractError("failed to pin the training source commit")
    venv = work / "venv"
    if not (venv / "bin/python").is_file():
        requested_python = str(manifest.get("runtime", {}).get("python", ""))
        if requested_python and tuple(requested_python.split(".")[:2]) != tuple(sys.version.split(".")[:2]):
            raise ContractError(
                f"Deep Learning image Python {sys.version.split()[0]} differs from E25 runtime {requested_python}"
            )
        base.run_command([sys.executable, "-m", "venv", str(venv)])
        packages = manifest.get("runtime", {}).get("packages", {})
        if packages.get("torch"):
            torch_version = str(packages["torch"]).split("+", 1)[0]
            cuda_digits = str(manifest.get("runtime", {}).get("cuda") or "").replace(".", "")
            install = [str(venv / "bin/pip"), "install", f"torch=={torch_version}"]
            if cuda_digits:
                install.extend(["--index-url", f"https://download.pytorch.org/whl/cu{cuda_digits}"])
            base.run_command(install)
        pins = [f"{name}=={version}" for name, version in packages.items() if name != "torch"]
        base.run_command([str(venv / "bin/pip"), "install", *pins, "-e", f"{repo}[torch,bacteria-titan]"])
    probe = base.run_command([
        str(venv / "bin/python"), "-c",
        "import json,torch,numpy,pandas,pyarrow; assert torch.cuda.is_available(); n=torch.cuda.get_device_name(0); assert 'A100' in n.upper(); print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':numpy.__version__,'pandas':pandas.__version__,'pyarrow':pyarrow.__version__,'gpu':n}))",
    ], capture=True)
    actual = json.loads(probe.stdout)
    expected = manifest.get("runtime", {}).get("packages", {})
    mismatch = {name: (expected[name], actual.get(name)) for name in expected if name in actual and expected[name] != actual[name]}
    if mismatch:
        raise ContractError(f"worker runtime differs from the E25 manifest: {mismatch}")
    return repo, venv / "bin/python"


def _begin_runtime_session(work: Path, max_total_hours: float) -> tuple[Path, float]:
    ledger_path = work / "CONTROL_RUNTIME.json"
    ledger: dict[str, Any] = {"format_version": 1, "consumed_hours": 0.0, "sessions": []}
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text())
    sessions = ledger.setdefault("sessions", [])
    if sessions and sessions[-1].get("state") == "running":
        # A hard 36-hour VM stop cannot run a finalizer. Count the previous
        # session's entire declared allowance so a restart can never undercount.
        sessions[-1]["state"] = "interrupted"
        sessions[-1]["elapsed_hours"] = float(sessions[-1]["planned_hours"])
        ledger["consumed_hours"] = float(ledger.get("consumed_hours", 0)) + float(sessions[-1]["planned_hours"])
    remaining = max_total_hours - float(ledger.get("consumed_hours", 0))
    if remaining <= 0:
        atomic_json(ledger_path, ledger)
        raise ContractError("cumulative standard-A100 runtime allowance is exhausted")
    planned = min(remaining, MAX_SESSION_HOURS - 0.25)
    sessions.append({"state": "running", "started_at": utc_now(), "planned_hours": planned})
    atomic_json(ledger_path, ledger)
    return ledger_path, planned


def _finish_runtime_session(ledger_path: Path, elapsed_hours: float, state: str) -> None:
    ledger = json.loads(ledger_path.read_text())
    session = ledger["sessions"][-1]
    if session.get("state") != "running":
        raise ContractError("runtime ledger lost its active session")
    charged = min(float(session["planned_hours"]), max(0.0, elapsed_hours))
    session.update(state=state, ended_at=utc_now(), elapsed_hours=charged)
    ledger["consumed_hours"] = float(ledger.get("consumed_hours", 0)) + charged
    atomic_json(ledger_path, ledger)


def _schedule_poweroff() -> None:
    # Delay gives systemd enough time to flush the service result and logs.
    subprocess.Popen(["shutdown", "-h", "+1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_worker(config_path: Path) -> int:
    config = json.loads(config_path.read_text())
    work = config_path.parent
    bundle = work / "input"
    if not (bundle / "TRAILBLAZER_INPUT_MANIFEST.json").is_file():
        bundle.mkdir(parents=True, exist_ok=True)
        base.run_command(["gcloud", "storage", "rsync", "--recursive", config["input_uri"], str(bundle)])
    curc.verify_input(bundle)
    manifest = json.loads((bundle / "TRAILBLAZER_INPUT_MANIFEST.json").read_text())
    input_bytes = sum(int(row["bytes"]) for row in manifest["files"].values())
    if shutil.disk_usage(work).free < input_bytes // 2 + 20 * 1024**3:
        raise ContractError("persistent control disk has insufficient free capacity")
    import torch
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0).upper():
        raise ContractError("worker does not expose an NVIDIA A100")
    repo, python_bin = _ensure_runtime(work, manifest)
    run_dir = work / "authoritative" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "latest.pt").is_file():
        verify_control_checkpoint(run_dir / "latest.pt")
    phase = str(config["phase"])
    command = training_command(bundle, run_dir, python_bin, pilot=phase == "pilot")
    ledger_path, session_hours = _begin_runtime_session(work, float(config["max_total_hours"]))
    start = time.monotonic()
    process = subprocess.Popen(command, cwd=repo)
    timed_out = False
    while process.poll() is None:
        time.sleep(60)
        _backup_checkpoint(run_dir, config)
        if (time.monotonic() - start) / 3600 >= session_hours:
            process.terminate()
            timed_out = True
            break
    return_code = process.wait(timeout=300)
    backup = _backup_checkpoint(run_dir, config)
    elapsed = time.monotonic() - start
    if phase == "pilot" and return_code == 0:
        report = pilot_report(run_dir, elapsed)
        atomic_json(run_dir / "GCP_PILOT_REPORT.json", report)
        base.run_command(["gcloud", "storage", "cp", str(run_dir / "GCP_PILOT_REPORT.json"),
                          f"gs://{config['bucket']}/{config['prefix']}/pilot/GCP_PILOT_REPORT.json"])
        _finish_runtime_session(ledger_path, elapsed / 3600, "pilot_completed")
        _schedule_poweroff()
    elif phase == "production" and return_code == 0:
        architecture = python_bin.parent / "seqtrainer-titans-stage-c-architecture"
        resume_verify = python_bin.parent / "seqtrainer-titans-stage-c-resume-verify"
        base.run_command([str(architecture), "--checkpoint", str(run_dir / "latest.pt"),
                          "--output-dir", str(run_dir)], cwd=repo)
        base.run_command([
            str(resume_verify), "--dataset-dir", str(bundle / "dataset"),
            "--panel-manifest", str(bundle / "panels/e25.json"),
            "--checkpoint", str(run_dir / "latest.pt"),
            "--output", str(run_dir / "resume_verification.json"), "--device", "cuda",
        ], cwd=repo)
        completed = json.loads((run_dir / "run_manifest.json").read_text())
        if not completed.get("scheduler_exhausted") or completed.get("stop_reason") != "panel_exhausted":
            raise ContractError("no-memory E25 exited without panel completion")
        complete = {"format_version": 1, "completed_at": utc_now(), "run_name": RUN_NAME,
                    "checkpoint": backup, "scheduler_exhausted": True, "stop_reason": "panel_exhausted"}
        atomic_json(run_dir / "COMPLETE.json", complete)
        base.run_command(["gcloud", "storage", "rsync", "--recursive", str(run_dir),
                          f"gs://{config['bucket']}/{config['prefix']}/complete"])
        _finish_runtime_session(ledger_path, elapsed / 3600, "completed")
        _schedule_poweroff()
    elif timed_out:
        atomic_json(run_dir / "PAUSED.json", {
            "paused_at": utc_now(), "reason": "cumulative_runtime_session_boundary",
            "checkpoint": backup,
        })
        base.run_command(["gcloud", "storage", "rsync", "--recursive", str(run_dir),
                          f"gs://{config['bucket']}/{config['prefix']}/paused"])
        _finish_runtime_session(ledger_path, elapsed / 3600, "paused")
        _schedule_poweroff()
    elif not timed_out:
        atomic_json(run_dir / "FAILED.json", {"failed_at": utc_now(), "return_code": return_code})
        base.run_command(["gcloud", "storage", "rsync", "--recursive", str(run_dir),
                          f"gs://{config['bucket']}/{config['prefix']}/failed"])
        _finish_runtime_session(ledger_path, elapsed / 3600, "failed")
        return return_code or 2
    return 0


def status(args: argparse.Namespace) -> dict[str, Any]:
    instance = base.json_command(["gcloud", "compute", "instances", "describe", args.name,
                                  "--project", args.project, "--zone", args.zone, "--format=json"])
    objects = base.run_command(["gcloud", "storage", "ls", "--recursive",
                                f"gs://{args.bucket}/{args.prefix}/**"], capture=True, check=False)
    return {"checked_at": utc_now(), "instance_state": instance.get("status"),
            "zone": args.zone, "objects": objects.stdout.splitlines()[-30:]}


def repatriate(args: argparse.Namespace) -> dict[str, Any]:
    destination = args.drive_root / "runs" / RUN_NAME
    if destination.exists():
        raise ContractError(f"refusing to overwrite existing Drive control: {destination}")
    with tempfile.TemporaryDirectory() as temporary:
        downloaded = Path(temporary) / "complete"
        downloaded.mkdir()
        base.run_command(["gcloud", "storage", "rsync", "--recursive",
                          f"gs://{args.bucket}/{args.prefix}/complete", str(downloaded)])
        complete = json.loads((downloaded / "COMPLETE.json").read_text())
        run_manifest = json.loads((downloaded / "run_manifest.json").read_text())
        if not complete.get("scheduler_exhausted") or complete.get("stop_reason") != "panel_exhausted":
            raise ContractError("Cloud control is incomplete")
        if not run_manifest.get("scheduler_exhausted") or run_manifest.get("stop_reason") != "panel_exhausted":
            raise ContractError("Cloud run manifest is incomplete")
        for required in ("MODEL_ARCHITECTURE.txt", "resume_verification.json"):
            if not (downloaded / required).is_file():
                raise ContractError(f"completed control is missing {required}")
        metadata = verify_control_checkpoint(downloaded / "latest.pt")
        partial = destination.with_name(destination.name + ".partial")
        if partial.exists():
            raise ContractError(f"stale partial Drive publication exists: {partial}")
        shutil.copytree(downloaded, partial)
        if curc.sha256_file(partial / "latest.pt") != curc.sha256_file(downloaded / "latest.pt"):
            raise ContractError("Drive control checkpoint checksum mismatch")
        os.replace(partial, destination)
    return {"published": str(destination), "optimizer_step": metadata["optimizer_step"],
            "processed_bases": metadata["processed_bases"]}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project", required=True)
    common.add_argument("--bucket", required=True)
    common.add_argument("--bundle", type=Path, required=True)
    common.add_argument("--region", default="auto")
    common.add_argument("--phase", choices=("pilot", "production"), required=True)
    common.add_argument("--pilot-report", type=Path)
    common.add_argument("--pilot-hours", type=float, default=2.0)
    common.add_argument("--hourly-cost-usd", type=float, default=DEFAULT_HOURLY_USD)
    common.add_argument("--storage-allowance-usd", type=float, default=DEFAULT_STORAGE_USD)
    common.add_argument("--fast", action="store_true")
    sub.add_parser("preflight", parents=[common], help="verify gate, bundle, quota, and cost")
    provision_cmd = sub.add_parser("provision", parents=[common], help="create or resume the private A100 control worker")
    provision_cmd.add_argument("--name", default=DEFAULT_NAME)
    provision_cmd.add_argument("--prefix", default=DEFAULT_PREFIX)
    provision_cmd.add_argument("--image-family", default="common-cu129-ubuntu-2204-nvidia-580")
    run = sub.add_parser("run", help="system-service worker entry point")
    run.add_argument("--config", type=Path, required=True)
    stat = sub.add_parser("status", help="show instance and immutable object status")
    stat.add_argument("--project", required=True)
    stat.add_argument("--bucket", required=True)
    stat.add_argument("--zone", required=True)
    stat.add_argument("--name", default=DEFAULT_NAME)
    stat.add_argument("--prefix", default=DEFAULT_PREFIX)
    rep = sub.add_parser("repatriate", help="publish a completed control atomically to Drive")
    rep.add_argument("--bucket", required=True)
    rep.add_argument("--prefix", default=DEFAULT_PREFIX)
    rep.add_argument("--drive-root", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "preflight":
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
    except (ContractError, curc.ContractError, base.ContractError, FileNotFoundError,
            subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
