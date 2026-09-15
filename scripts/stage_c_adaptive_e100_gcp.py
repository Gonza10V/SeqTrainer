#!/usr/bin/env python3
"""Grant-gated GCP runner for adaptive E100 failover or independent replication.

``gcp-failover`` continues seed 20260751 from an immutable CURC recovery bundle.
``gcp-replica`` trains independent seed 20260752 through E25 and E100.  The two
roles never share an authoritative run directory or checkpoint namespace.
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


base = _load("stage_c_adaptive_base", ROOT / "stage_c_c19_gcp.py")
curc = _load("stage_c_adaptive_curc", ROOT / "stage_c_c20_curc.py")
control = _load("stage_c_adaptive_control", ROOT / "stage_c_e25_control_gcp.py")

ROLE_FAILOVER = "gcp-failover"
ROLE_REPLICA = "gcp-replica"
ROLES = (ROLE_FAILOVER, ROLE_REPLICA)
DEFAULT_MACHINE = "a2-highgpu-1g"
DEFAULT_PREFIX = "stage-c-adaptive-e100"
DEFAULT_NAME = "stage-c-adaptive-e100"
DEFAULT_DISK_GB = 200
DEFAULT_HOURLY_USD = 3.673385
GRANT_BUDGET_USD = 5_000.0
GRANT_GPU_HOURS = 1_250.0
MAX_SESSION_HOURS = 36.0

SYSTEMD_UNIT = """[Unit]
Description=Stage C adaptive E100 GCP trajectory
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=root
WorkingDirectory=/mnt/disks/stage-c-adaptive
ExecStart=/usr/bin/python3 /usr/local/sbin/stage_c_adaptive_e100_gcp.py run --config /mnt/disks/stage-c-adaptive/ADAPTIVE_GCP_CONFIG.json
Restart=no
TimeoutStartSec=infinity

[Install]
WantedBy=multi-user.target
"""


class ContractError(RuntimeError):
    """A grant, role, checkpoint, runtime, or spending contract failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def role_contract(
    bundle: Path, role: str, recovery: Path | None = None,
) -> dict[str, Any]:
    manifest = curc.verify_input(bundle)
    if role not in ROLES:
        raise ContractError(f"role must be one of {ROLES}")
    if role == ROLE_FAILOVER:
        if recovery is None:
            raise ContractError("gcp-failover requires a CURC recovery bundle")
        recovery_manifest = curc.verify_recovery(recovery)
        parent_step = int(manifest["parent"]["optimizer_step"])
        recovery_step = int(recovery_manifest["checkpoint"]["optimizer_step"])
        if recovery_step < parent_step:
            raise ContractError("CURC recovery regresses behind the completed E25 parent")
        return {
            "role": role, "scientific_seed": curc.PRIMARY_SEED,
            "trajectory_id": f"adaptive-seed-{curc.PRIMARY_SEED}",
            "input": manifest, "recovery": recovery_manifest,
        }
    if recovery is not None:
        raise ContractError("gcp-replica must not inherit the primary trajectory recovery")
    return {
        "role": role, "scientific_seed": curc.REPLICA_SEED,
        "trajectory_id": f"adaptive-seed-{curc.REPLICA_SEED}",
        "input": manifest, "recovery": None,
    }


def verify_credit_activation(path: Path, project: str, role: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError("production requires a private credit-activation JSON file")
    payload = json.loads(path.read_text())
    if payload.get("active") is not True or payload.get("project") != project:
        raise ContractError("research credit is not confirmed active for this project")
    if float(payload.get("award_usd", 0)) < GRANT_BUDGET_USD:
        raise ContractError("activation record does not confirm the planned USD 5,000 award")
    if not payload.get("award_id") or not payload.get("billing_account_id"):
        raise ContractError("activation record lacks award or billing-account identity")
    allocations = payload.get("gpu_hour_allocations")
    if not isinstance(allocations, Mapping) or any(name not in ROLES for name in allocations):
        raise ContractError("activation record lacks role-specific adaptive GPU-hour allocations")
    try:
        hours = {name: float(value) for name, value in allocations.items()}
    except (TypeError, ValueError) as error:
        raise ContractError("credit GPU-hour allocations are invalid") from error
    if any(value <= 0 for value in hours.values()) or sum(hours.values()) > GRANT_GPU_HOURS:
        raise ContractError("credit GPU-hour allocations exceed the 1,250-hour campaign cap")
    if role is not None and role not in hours:
        raise ContractError(f"activation record does not authorize role {role}")
    payload["gpu_hour_allocations"] = hours
    return payload


def verify_failover_authorization(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError("production failover requires a private failover-authorization JSON file")
    payload = json.loads(path.read_text())
    if payload.get("authorized") is not True:
        raise ContractError("GCP failover is not authorized")
    reason = payload.get("reason")
    if reason not in {"curc_not_started_14_days", "curc_nonrecoverable_failure"}:
        raise ContractError("failover reason is not an allowed trigger")
    if not payload.get("authorized_by") or not payload.get("authorized_at"):
        raise ContractError("failover authorization lacks identity or timestamp")
    if reason == "curc_not_started_14_days":
        try:
            planned = datetime.fromisoformat(str(payload["curc_planned_start_at"]))
            authorized = datetime.fromisoformat(str(payload["authorized_at"]))
        except (KeyError, ValueError) as error:
            raise ContractError("14-day failover authorization has invalid timestamps") from error
        if planned.tzinfo is None or authorized.tzinfo is None:
            raise ContractError("failover timestamps must be timezone-aware")
        if (authorized - planned).total_seconds() < 14 * 24 * 3600:
            raise ContractError("CURC 14-day start window has not elapsed")
    elif not payload.get("failure_evidence"):
        raise ContractError("non-recoverable CURC failure requires an evidence reference")
    return payload


def _run_id(role: str, stage: str) -> str:
    if role == ROLE_FAILOVER:
        return "medium_adaptive_gcp_failover_chunk_v1"
    return {
        "e25": "medium_adaptive_seed2_e25_v1",
        "e50": "medium_adaptive_e50_milestone_v1",
        "e75": "medium_adaptive_e75_milestone_v1",
        "e100": "medium_adaptive_seed2_e100_increment_v1",
    }[stage]


def training_command(
    bundle: Path, run_root: Path, python_bin: Path, *, role: str, stage: str,
    pilot_target_step: int | None = None,
) -> list[str]:
    contract = role_contract(bundle, role, None if role == ROLE_REPLICA else run_root / "recovery")
    if stage not in {"e25", "e50", "e75", "e100"}:
        raise ContractError("adaptive GCP stage must be e25, e50, e75, or e100")
    if role == ROLE_FAILOVER and stage == "e25":
        raise ContractError("failover cannot restart the completed primary E25 run")
    seed = int(contract["scientific_seed"])
    qualification = json.loads((bundle / "qualification/qualification_selection.json").read_text())
    e25 = stage == "e25"
    run_dir = run_root / ("e25" if e25 else "e100")
    panel = bundle / "panels" / ("e25.json" if e25 else "e100_additions.json")
    executable = str(python_bin.parent / "seqtrainer-titans-stage-c-train")
    command = [
        executable, "--dataset-dir", str(bundle / "dataset"),
        "--panel-manifest", str(panel),
        "--validation-panel-manifest", str(bundle / "panels/validation.json"),
        "--run-dir", str(run_dir), "--memory-mode", "adaptive",
        "--horizon", "3", "--batch-size", str(qualification["batch_size"]),
        "--seed", str(seed), "--scheduler-policy", "stateful_rotation",
        "--scheduler-burst-segments", "96", "--checkpoint-every", "250",
        "--learning-rate", "3e-5", "--min-learning-rate", "3e-6",
        "--lr-warmup-bases", "2000000", "--lr-decay-bases", "100000000",
        "--weight-decay", "0.1", "--gradient-clip-norm", "0.5",
        "--activation", str(qualification["activation"]), "--block-count", "12",
        "--d-model", "256", "--num-heads", "8", "--persistent-tokens", "4",
        "--validation-streams", "1", "--validation-segments", "1",
        *curc.DEEP_FLAGS, "--protocol", str(bundle / "study/protocol.json"),
        "--protocol-amendment", str(bundle / f"study/{curc.AMENDMENT_NAME}"),
        "--run-id", _run_id(role, stage),
    ]
    if role == ROLE_REPLICA and not e25 and not (run_dir / "latest.pt").is_file():
        parent = run_root / "e25/latest.pt"
        if not parent.is_file():
            raise ContractError("seed-2 E100 requires its own completed E25 checkpoint")
        command.extend(["--warm-start-checkpoint", str(parent), "--no-resume"])
    if pilot_target_step is not None:
        command.extend(["--max-optimizer-steps", str(pilot_target_step)])
    elif stage == "e50":
        command.extend(["--max-valid-bases", str(curc.E50_BASES)])
    elif stage == "e75":
        command.extend(["--max-valid-bases", str(curc.E75_BASES)])
    else:
        command.append("--require-panel-completion")
    return command


def stage_recovery(recovery: Path, run_root: Path) -> dict[str, Any]:
    manifest = curc.verify_recovery(recovery)
    destination = run_root / "e100"
    destination.mkdir(parents=True, exist_ok=True)
    source = recovery / manifest["checkpoint"]["relative_path"]
    current = destination / "latest.pt"
    if current.is_file():
        current_meta = curc._verify_adaptive_checkpoint(current, curc.PRIMARY_SEED)
        if current_meta["optimizer_step"] < manifest["checkpoint"]["optimizer_step"]:
            raise ContractError("existing GCP state is older than the requested recovery; inspect before replacement")
        return {"state": "kept_newer_or_equal", "checkpoint": current_meta}
    for relative in manifest["files"]:
        item = recovery / relative
        curc.stable_copy(item, destination / relative)
    if curc.sha256_file(current) != curc.sha256_file(source):
        raise ContractError("staged failover checkpoint checksum mismatch")
    return {"state": "staged", "checkpoint": manifest["checkpoint"]}


def next_stage(role: str, run_root: Path) -> str:
    if role == ROLE_REPLICA:
        e25_manifest = run_root / "e25/run_manifest.json"
        if not e25_manifest.is_file():
            return "e25"
        completed = json.loads(e25_manifest.read_text())
        if not completed.get("scheduler_exhausted") or completed.get("stop_reason") != "panel_exhausted":
            return "e25"
    latest = run_root / "e100/latest.pt"
    if not latest.is_file():
        return "e50"
    metadata = curc.checkpoint_metadata(latest)
    if metadata["processed_bases"] < curc.E50_BASES:
        return "e50"
    if metadata["processed_bases"] < curc.E75_BASES:
        return "e75"
    return "e100"


def _upload_tree_immutable(source: Path, bucket: str, prefix: str, category: str) -> str:
    manifest_name = "MILESTONE_MANIFEST.json" if category == "milestones" else "RECOVERY_MANIFEST.json"
    manifest = json.loads((source / manifest_name).read_text())
    uri = f"gs://{bucket}/{prefix}/{category}/{manifest['bundle_id']}"
    marker = f"{uri}/PUBLISHED_{manifest_name}"
    if base.run_command(["gcloud", "storage", "ls", marker], capture=True, check=False).returncode == 0:
        raise ContractError(f"immutable GCS publication already exists: {uri}")
    base.run_command(["gcloud", "storage", "rsync", "--recursive", str(source), uri])
    base.upload_immutable(source / manifest_name, marker)
    return uri


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    contract = role_contract(args.bundle, args.role, args.recovery)
    activation = None
    failover_authorization = None
    if args.phase == "production":
        if args.credit_activation is None:
            raise ContractError("production preflight requires --credit-activation")
        activation = verify_credit_activation(args.credit_activation, args.project, args.role)
        if args.role == ROLE_FAILOVER:
            if args.failover_authorization is None:
                raise ContractError("production failover requires --failover-authorization")
            failover_authorization = verify_failover_authorization(args.failover_authorization)
    billing = base.json_command(["gcloud", "billing", "projects", "describe", args.project, "--format=json"])
    if not billing.get("billingEnabled"):
        raise ContractError("billing is not enabled")
    base.run_command(["gcloud", "storage", "ls", f"gs://{args.bucket}"])
    regions = base.quota_regions(args.project)
    if args.region != "auto":
        regions = [region for region in regions if region == args.region]
    if not regions or not base.global_gpu_available(args.project):
        raise ContractError("regional NVIDIA_A100_GPUS and global GPU quota are required")
    input_bytes = sum(int(row["bytes"]) for row in contract["input"]["files"].values())
    if contract["recovery"]:
        input_bytes += sum(int(row["bytes"]) for row in contract["recovery"]["files"].values())
    disk_gb = max(DEFAULT_DISK_GB, int(input_bytes * 1.5 / 1_000_000_000) + 1)
    max_hours = (
        args.pilot_hours if args.phase == "pilot"
        else float(activation["gpu_hour_allocations"][args.role])
    )
    projected = max_hours * args.hourly_cost_usd + args.storage_allowance_usd
    if args.phase == "production" and projected > GRANT_BUDGET_USD:
        raise ContractError("planned adaptive GCP cost exceeds the USD 5,000 award")
    return {
        "format_version": 1, "checked_at": utc_now(), "project": args.project,
        "bucket": args.bucket, "role": args.role, "phase": args.phase,
        "trajectory_id": contract["trajectory_id"], "scientific_seed": contract["scientific_seed"],
        "selected_region": regions[0], "eligible_regions": regions,
        "machine_type": DEFAULT_MACHINE, "gpu": "NVIDIA A100 40 GB",
        "disk_gb": disk_gb, "max_total_hours": max_hours,
        "estimated_total_usd": projected, "billing_account": billing.get("billingAccountName"),
        "credit_award_id": activation.get("award_id") if activation else None,
        "failover_reason": failover_authorization.get("reason") if failover_authorization else None,
    }


def _upload_input(bundle: Path, bucket: str, prefix: str) -> str:
    manifest = curc.verify_input(bundle)
    uri = f"gs://{bucket}/{prefix}/inputs/{manifest['bundle_id']}"
    marker = f"{uri}/PUBLISHED_TRAILBLAZER_INPUT_MANIFEST.json"
    if base.run_command(["gcloud", "storage", "ls", marker], capture=True, check=False).returncode:
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
    suffix = "failover" if args.role == ROLE_FAILOVER else "replica"
    name = args.name or f"{DEFAULT_NAME}-{suffix}"
    prefix = f"{args.prefix}/{suffix}"
    service_account = f"{name}@{args.project}.iam.gserviceaccount.com"
    input_uri = _upload_input(args.bundle, args.bucket, prefix)
    recovery_uri = None
    if args.recovery:
        recovery_uri = _upload_tree_immutable(args.recovery, args.bucket, prefix, "recovery")
    billing_account = str(report.get("billing_account") or "").rsplit("/", 1)[-1]
    project_info = base.json_command(["gcloud", "projects", "describe", args.project, "--format=json"])
    budget_name = "Stage C research credits USD 5000"
    existing_budgets = base.json_command([
        "gcloud", "billing", "budgets", "list", "--billing-account", billing_account,
        "--filter", f"displayName={budget_name}", "--format=json",
    ])
    if not existing_budgets:
        base.run_command([
            "gcloud", "billing", "budgets", "create", "--billing-account", billing_account,
            "--display-name", budget_name, "--budget-amount", "5000USD",
            "--filter-projects", f"projects/{project_info['projectId']}",
            "--threshold-rule", "percent=0.25", "--threshold-rule", "percent=0.5",
            "--threshold-rule", "percent=0.75", "--threshold-rule", "percent=0.9",
            "--threshold-rule", "percent=1.0",
        ])
    config = {
        "project": args.project, "bucket": args.bucket, "prefix": prefix,
        "name": name, "zone": zone, "role": args.role, "phase": args.phase,
        "input_uri": input_uri, "recovery_uri": recovery_uri,
        "max_total_hours": report["max_total_hours"],
    }
    commands = [
        ["gcloud", "iam", "service-accounts", "create", name, "--project", args.project,
         "--display-name", f"Stage C adaptive E100 {suffix}"],
        ["gcloud", "projects", "add-iam-policy-binding", args.project,
         "--member", f"serviceAccount:{service_account}", "--role", "roles/storage.objectAdmin"],
        ["gcloud", "projects", "add-iam-policy-binding", args.project,
         "--member", f"serviceAccount:{service_account}", "--role", "roles/logging.logWriter"],
        ["gcloud", "compute", "networks", "create", name, "--project", args.project, "--subnet-mode", "custom"],
        ["gcloud", "compute", "networks", "subnets", "create", name, "--project", args.project,
         "--network", name, "--region", region, "--range", "10.73.0.0/24", "--enable-private-ip-google-access"],
        ["gcloud", "compute", "firewall-rules", "create", f"{name}-iap-ssh", "--project", args.project,
         "--network", name, "--direction", "INGRESS", "--action", "ALLOW", "--rules", "tcp:22",
         "--source-ranges", "35.235.240.0/20"],
        ["gcloud", "compute", "routers", "create", name, "--project", args.project,
         "--network", name, "--region", region],
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
        config_path = temp / "ADAPTIVE_GCP_CONFIG.json"
        atomic_json(config_path, config)
        support = {
            Path(__file__): "stage_c_adaptive_e100_gcp.py",
            ROOT / "stage_c_c20_curc.py": "stage_c_c20_curc.py",
            ROOT / "stage_c_c19_gcp.py": "stage_c_c19_gcp.py",
            ROOT / "stage_c_e25_control_gcp.py": "stage_c_e25_control_gcp.py",
            config_path: "ADAPTIVE_GCP_CONFIG.json",
        }
        uris: dict[str, str] = {}
        for source, target in support.items():
            uri = f"gs://{args.bucket}/{prefix}/control/{target}"
            base.run_command(["gcloud", "storage", "cp", str(source), uri])
            uris[target] = uri
        startup = f"""#!/bin/bash
set -euo pipefail
DISK=/dev/disk/by-id/google-stage-c-adaptive
MOUNT=/mnt/disks/stage-c-adaptive
mkdir -p $MOUNT
if ! blkid $DISK; then mkfs.ext4 -m 0 -F $DISK; fi
mountpoint -q $MOUNT || mount -o discard,defaults $DISK $MOUNT
gcloud storage cp {uris['stage_c_adaptive_e100_gcp.py']} /usr/local/sbin/stage_c_adaptive_e100_gcp.py
gcloud storage cp {uris['stage_c_c20_curc.py']} /usr/local/sbin/stage_c_c20_curc.py
gcloud storage cp {uris['stage_c_c19_gcp.py']} /usr/local/sbin/stage_c_c19_gcp.py
gcloud storage cp {uris['stage_c_e25_control_gcp.py']} /usr/local/sbin/stage_c_e25_control_gcp.py
gcloud storage cp {uris['ADAPTIVE_GCP_CONFIG.json']} $MOUNT/ADAPTIVE_GCP_CONFIG.json
chmod 755 /usr/local/sbin/stage_c_*.py
cat > /etc/systemd/system/stage-c-adaptive.service <<'ADAPTIVEUNIT'
{SYSTEMD_UNIT}ADAPTIVEUNIT
systemctl daemon-reload
systemctl enable --now stage-c-adaptive.service
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
            "--disk", f"name={name},device-name=stage-c-adaptive,mode=rw,boot=no,auto-delete=no",
            "--metadata-from-file", f"startup-script={startup_path}",
        ], capture=True, check=False)
        if create.returncode and "already exists" not in (create.stdout + create.stderr).lower():
            raise subprocess.CalledProcessError(create.returncode, create.args, create.stdout, create.stderr)
        if create.returncode:
            base.run_command(["gcloud", "compute", "instances", "start", name,
                              "--project", args.project, "--zone", zone])
    return {**report, "instance": name, "zone": zone, "prefix": prefix,
            "input_uri": input_uri, "recovery_uri": recovery_uri}


def _backup(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any] | None:
    source = run_dir / "latest.pt"
    if not source.is_file():
        return None
    seed = curc.PRIMARY_SEED if config["role"] == ROLE_FAILOVER else curc.REPLICA_SEED
    metadata = curc._verify_adaptive_checkpoint(source, seed)
    digest = curc.sha256_file(source)
    marker = run_dir / ".last_uploaded_checkpoint.json"
    if marker.is_file():
        prior = json.loads(marker.read_text())
        if int(prior["optimizer_step"]) > metadata["optimizer_step"]:
            raise ContractError("adaptive GCP checkpoint regressed")
        if prior.get("sha256") == digest:
            return prior
    snapshot = run_dir / ".checkpoint_upload.pt"
    curc.stable_copy(source, snapshot)
    uri = (f"gs://{config['bucket']}/{config['prefix']}/checkpoints/"
           f"step-{metadata['optimizer_step']:09d}-{digest}/latest.pt")
    base.upload_immutable(snapshot, uri)
    snapshot.unlink(missing_ok=True)
    record = {"optimizer_step": metadata["optimizer_step"], "processed_bases": metadata["processed_bases"],
              "sha256": digest, "uri": uri, "uploaded_at": utc_now()}
    atomic_json(marker, record)
    return record


def run_worker(config_path: Path) -> int:
    config = json.loads(config_path.read_text())
    work = config_path.parent
    bundle = work / "input"
    if not (bundle / "TRAILBLAZER_INPUT_MANIFEST.json").is_file():
        bundle.mkdir(parents=True, exist_ok=True)
        base.run_command(["gcloud", "storage", "rsync", "--recursive", config["input_uri"], str(bundle)])
    recovery = None
    if config["role"] == ROLE_FAILOVER:
        recovery = work / "recovery"
        if not (recovery / "RECOVERY_MANIFEST.json").is_file():
            recovery.mkdir(parents=True, exist_ok=True)
            base.run_command(["gcloud", "storage", "rsync", "--recursive", config["recovery_uri"], str(recovery)])
    contract = role_contract(bundle, config["role"], recovery)
    repo, python_bin = control._ensure_runtime(work, contract["input"])
    run_root = work / "authoritative" / contract["trajectory_id"]
    if recovery:
        staged = run_root / "recovery"
        if not staged.exists():
            shutil.copytree(recovery, staged)
        stage_recovery(staged, run_root)
    stage = next_stage(config["role"], run_root)
    run_dir = run_root / ("e25" if stage == "e25" else "e100")
    run_dir.mkdir(parents=True, exist_ok=True)
    pilot_target = None
    if config["phase"] == "pilot":
        start_step = curc.checkpoint_metadata(run_dir / "latest.pt")["optimizer_step"] if (run_dir / "latest.pt").is_file() else 0
        pilot_target = start_step + 1
    command = training_command(
        bundle, run_root, python_bin, role=config["role"], stage=stage,
        pilot_target_step=pilot_target,
    )
    ledger_path, session_hours = control._begin_runtime_session(work, float(config["max_total_hours"]))
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=repo)
    timed_out = False
    while process.poll() is None:
        time.sleep(60)
        _backup(run_dir, config)
        if (time.monotonic() - started) / 3600 >= min(session_hours, MAX_SESSION_HOURS - 0.25):
            process.terminate()
            timed_out = True
            break
    return_code = process.wait(timeout=300)
    backup = _backup(run_dir, config)
    elapsed_hours = (time.monotonic() - started) / 3600
    if return_code == 0:
        metadata = curc._verify_adaptive_checkpoint(run_dir / "latest.pt", contract["scientific_seed"])
        if config["phase"] == "pilot":
            if metadata["optimizer_step"] != pilot_target:
                raise ContractError("adaptive GCP pilot did not advance exactly one optimizer step")
            report = {"format_version": 1, "completed_at": utc_now(), "role": config["role"],
                      "trajectory_id": contract["trajectory_id"], "scientific_seed": contract["scientific_seed"],
                      "optimizer_step": metadata["optimizer_step"], "processed_bases": metadata["processed_bases"]}
            atomic_json(run_dir / "GCP_ADAPTIVE_PILOT_REPORT.json", report)
            base.run_command(["gcloud", "storage", "cp", str(run_dir / "GCP_ADAPTIVE_PILOT_REPORT.json"),
                              f"gs://{config['bucket']}/{config['prefix']}/pilot/GCP_ADAPTIVE_PILOT_REPORT.json"])
            control._finish_runtime_session(ledger_path, elapsed_hours, "pilot_completed")
        else:
            milestone = stage
            package_root = work / "publications"
            milestone_manifest = curc.package_milestone(
                run_dir, package_root, milestone,
                trajectory_id=contract["trajectory_id"], execution_role=config["role"],
                scientific_seed=contract["scientific_seed"],
            )
            publication = package_root / milestone_manifest["bundle_id"]
            uri = _upload_tree_immutable(publication, config["bucket"], config["prefix"], "milestones")
            atomic_json(run_root / "LAST_MILESTONE.json", {"milestone": milestone, "uri": uri,
                                                            "checkpoint": backup})
            state = "completed" if milestone == "e100" else f"completed_{milestone}"
            control._finish_runtime_session(ledger_path, elapsed_hours, state)
    elif timed_out:
        atomic_json(run_root / "PAUSED.json", {"paused_at": utc_now(), "stage": stage,
                                               "reason": "runtime_session_boundary", "checkpoint": backup})
        base.run_command(["gcloud", "storage", "rsync", "--recursive", str(run_root),
                          f"gs://{config['bucket']}/{config['prefix']}/paused"])
        control._finish_runtime_session(ledger_path, elapsed_hours, "paused")
    else:
        atomic_json(run_root / "FAILED.json", {"failed_at": utc_now(), "stage": stage,
                                               "return_code": return_code, "checkpoint": backup})
        base.run_command(["gcloud", "storage", "rsync", "--recursive", str(run_root),
                          f"gs://{config['bucket']}/{config['prefix']}/failed"])
        control._finish_runtime_session(ledger_path, elapsed_hours, "failed")
        return return_code or 2
    control._schedule_poweroff()
    return 0


def status(args: argparse.Namespace) -> dict[str, Any]:
    instance = base.json_command(["gcloud", "compute", "instances", "describe", args.name,
                                  "--project", args.project, "--zone", args.zone, "--format=json"])
    objects = base.run_command(["gcloud", "storage", "ls", "--recursive",
                                f"gs://{args.bucket}/{args.prefix}/**"], capture=True, check=False)
    return {"checked_at": utc_now(), "instance_state": instance.get("status"),
            "zone": args.zone, "objects": objects.stdout.splitlines()[-40:]}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    contract = sub.add_parser("verify-contract", help="verify role, input, and optional recovery offline")
    contract.add_argument("--bundle", type=Path, required=True)
    contract.add_argument("--role", choices=ROLES, required=True)
    contract.add_argument("--recovery", type=Path)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project", required=True)
    common.add_argument("--bucket", required=True)
    common.add_argument("--bundle", type=Path, required=True)
    common.add_argument("--role", choices=ROLES, required=True)
    common.add_argument("--recovery", type=Path)
    common.add_argument("--region", default="auto")
    common.add_argument("--phase", choices=("pilot", "production"), required=True)
    common.add_argument("--credit-activation", type=Path)
    common.add_argument("--failover-authorization", type=Path)
    common.add_argument("--pilot-hours", type=float, default=2.0)
    common.add_argument("--hourly-cost-usd", type=float, default=DEFAULT_HOURLY_USD)
    common.add_argument("--storage-allowance-usd", type=float, default=408.0)
    sub.add_parser("preflight", parents=[common], help="verify billing, quota, award, and scientific contract")
    provision_cmd = sub.add_parser("provision", parents=[common], help="create the private adaptive A100 worker")
    provision_cmd.add_argument("--name")
    provision_cmd.add_argument("--prefix", default=DEFAULT_PREFIX)
    provision_cmd.add_argument("--image-family", default="common-cu129-ubuntu-2204-nvidia-580")
    command = sub.add_parser("training-command", help="print the exact command for review")
    command.add_argument("--bundle", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--python-bin", type=Path, required=True)
    command.add_argument("--role", choices=ROLES, required=True)
    command.add_argument("--stage", choices=("e25", "e50", "e75", "e100"), required=True)
    run = sub.add_parser("run", help="system-service worker entry point")
    run.add_argument("--config", type=Path, required=True)
    stat = sub.add_parser("status", help="show VM and immutable-object status")
    stat.add_argument("--project", required=True)
    stat.add_argument("--bucket", required=True)
    stat.add_argument("--zone", required=True)
    stat.add_argument("--name", required=True)
    stat.add_argument("--prefix", required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "verify-contract":
            result = role_contract(args.bundle, args.role, args.recovery)
        elif args.command == "preflight":
            result = preflight(args)
        elif args.command == "provision":
            result = provision(args)
        elif args.command == "training-command":
            result = {"command": training_command(
                args.bundle, args.run_root, args.python_bin,
                role=args.role, stage=args.stage,
            )}
        elif args.command == "run":
            return run_worker(args.config)
        elif args.command == "status":
            result = status(args)
        else:
            raise AssertionError(args.command)
    except (ContractError, curc.ContractError, control.ContractError, base.ContractError,
            FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
