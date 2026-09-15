#!/usr/bin/env python3
"""Checkpoint-safe CURC continuation for the Stage C adaptive E25-to-E100 trailblazer.

The launcher is intentionally independent of Google credentials.  Colab creates an
immutable input bundle, a laptop transfers it to Alpine, and this script verifies,
pilots, submits, monitors, and packages the continuation.  Milestone bundles return
through the laptop and Cloud Storage before a Colab notebook publishes them to Drive.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


TRAINING_COMMIT = "ae72fae21ff9a0b50e4fe1d9d642c38c643b4923"
STUDY = "stage_c_ecoli_medium_deep_memory_v3"
E25_RUN = "c19_v3_medium_adaptive_e25"
E100_RUN = "c20_v3_medium_adaptive_e100_increment"
E100_RUN_ID = "medium_adaptive_e100_increment_v1"
AMENDMENT_NAME = "adaptive_trailblazer_e100_v1.json"
PRIMARY_SEED = 20260751
REPLICA_SEED = 20260752
E50_BASES = 50_000_000
E75_BASES = 75_000_000
DEFAULT_THROUGHPUT = 53.9197280010886
DEFAULT_WALLTIME_HOURS = 160
GPU_BILLING_WEIGHT = 108.6
DEFAULT_CPU_COUNT = 8
CPU_BILLING_WEIGHT = 1.0
MILESTONES = {"e50": E50_BASES, "e75": E75_BASES}

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
    """A scientific, transfer, scheduler, or runtime contract failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while (chunk := handle.read(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def run_command(
    command: Sequence[str], *, cwd: Path | None = None, capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), cwd=cwd, text=True, capture_output=capture, check=check,
    )


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise ContractError("PyTorch is required to inspect an owned Stage C checkpoint") from error
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ContractError("checkpoint payload is not a mapping")
    trainer = payload.get("trainer_state")
    scheduler = payload.get("scheduler_state")
    model = payload.get("model_config")
    if not all(isinstance(item, Mapping) for item in (trainer, scheduler, model)):
        raise ContractError("checkpoint lacks trainer, scheduler, or model configuration")
    return {
        "optimizer_step": int(trainer.get("optimizer_step", -1)),
        "processed_bases": int(trainer.get("processed_bases", -1)),
        "processed_segments": int(trainer.get("processed_segments", -1)),
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
        "dataset_components": payload.get("dataset_components"),
        "code_commit": payload.get("code_commit"),
        "model_config": dict(model),
        "scheduler_policy": scheduler.get("policy"),
        "scheduler_batch_size": scheduler.get("batch_size"),
        "scheduler_burst_segments": scheduler.get("burst_segments"),
        "scheduler_seed": scheduler.get("seed"),
    }


def runtime_manifest() -> dict[str, Any]:
    packages: dict[str, str] = {}
    cuda = None
    for name in ("torch", "numpy", "pandas", "pyarrow"):
        try:
            packages[name] = str(__import__(name).__version__)
        except Exception:
            pass
    try:
        import torch
        cuda = torch.version.cuda
    except Exception:
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
    }


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from (item for item in sorted(path.rglob("*")) if item.is_file())


def stable_copy(source: Path, destination: Path) -> dict[str, Any]:
    before_stat = source.stat()
    before = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, partial)
    copied = sha256_file(partial)
    after = sha256_file(source)
    after_stat = source.stat()
    if before != copied or before != after or (
        before_stat.st_size, before_stat.st_mtime_ns
    ) != (after_stat.st_size, after_stat.st_mtime_ns):
        partial.unlink(missing_ok=True)
        raise ContractError(f"source changed during stable copy: {source}")
    os.replace(partial, destination)
    return {"sha256": before, "bytes": before_stat.st_size, "mtime_ns": before_stat.st_mtime_ns}


def _export_paths(drive_root: Path, repo_root: Path) -> list[tuple[Path, Path]]:
    study = drive_root / "study" / STUDY
    run = drive_root / "runs" / E25_RUN
    return [
        (drive_root / "stage_c_dataset/ordered_streams/nonoverlap_6mer_v1", Path("dataset")),
        (study / "panels/e25.json", Path("panels/e25.json")),
        (study / "panels/e100_additions.json", Path("panels/e100_additions.json")),
        (study / "panels/validation.json", Path("panels/validation.json")),
        (study / "panels/test.json", Path("panels/test.json")),
        (drive_root / "runs/c18_v3_medium_a100_qualification/qualification_selection.json", Path("qualification/qualification_selection.json")),
        (run / "latest.pt", Path("parent/latest.pt")),
        (run / "run_manifest.json", Path("parent/run_manifest.json")),
        (run / "LIVE_STATUS.json", Path("parent/LIVE_STATUS.json")),
        (run / "colab_run_manifest.json", Path("parent/colab_run_manifest.json")),
        (run / "scale_analysis_v1/gate/scale_gate.json", Path("parent/scale_gate.json")),
        (study / "protocol.json", Path("study/protocol.json")),
        (repo_root / f"studies/{STUDY}/amendments/{AMENDMENT_NAME}", Path(f"study/{AMENDMENT_NAME}")),
    ]


def export_input(drive_root: Path, repo_root: Path, output: Path) -> dict[str, Any]:
    parent = drive_root / "runs" / E25_RUN
    gate_path = parent / "scale_analysis_v1/gate/scale_gate.json"
    checkpoint = parent / "latest.pt"
    if not gate_path.is_file() or json.loads(gate_path.read_text()).get("proceed") is not True:
        raise ContractError("the frozen E25 scale gate does not authorize E100 continuation")
    run_manifest_path = parent / "run_manifest.json"
    live_path = parent / "LIVE_STATUS.json"
    if not run_manifest_path.is_file() or not live_path.is_file():
        raise ContractError("E25 completion manifests are missing")
    run_manifest = json.loads(run_manifest_path.read_text())
    live = json.loads(live_path.read_text())
    if (
        run_manifest.get("scheduler_exhausted") is not True
        or run_manifest.get("stop_reason") != "panel_exhausted"
        or live.get("state") != "completed"
    ):
        raise ContractError("E25 is not a completed, panel-exhausted run")
    metadata = checkpoint_metadata(checkpoint)
    if metadata["code_commit"] != TRAINING_COMMIT:
        raise ContractError("E25 checkpoint came from the wrong training commit")
    if metadata["model_config"].get("memory_mode") != "adaptive":
        raise ContractError("E25 parent is not the adaptive trailblazer")
    bundle_id = f"e25-step-{metadata['optimizer_step']:09d}-{sha256_file(checkpoint)[:16]}"
    bundle = output / bundle_id
    if bundle.exists():
        raise ContractError(f"immutable trailblazer input already exists: {bundle}")
    files: dict[str, Any] = {}
    try:
        for source_root, relative_root in _export_paths(drive_root, repo_root):
            if not source_root.exists():
                if source_root.name == "test.json":
                    continue
                raise ContractError(f"required trailblazer input is missing: {source_root}")
            for source in iter_files(source_root):
                relative = relative_root if source_root.is_file() else relative_root / source.relative_to(source_root)
                files[relative.as_posix()] = stable_copy(source, bundle / relative)
        e100_panel = json.loads((bundle / "panels/e100_additions.json").read_text())
        completion_bases = metadata["processed_bases"] + int(e100_panel["predictable_bases"])
        colab = json.loads((bundle / "parent/colab_run_manifest.json").read_text())
        environment = colab.get("environment")
        if not isinstance(environment, Mapping):
            raise ContractError("C19 Colab runtime manifest lacks its environment")
        raw_packages = environment.get("packages", {})
        runtime_packages = {
            name: str(version)
            for name, version in raw_packages.items()
            if name in {"torch", "numpy", "pandas", "pyarrow", "transformers", "tokenizers"}
            and version not in {None, "", "unavailable"}
        } if isinstance(raw_packages, Mapping) else {}
        if environment.get("torch"):
            runtime_packages["torch"] = str(environment["torch"])
        manifest = {
            "format_version": 1,
            "immutable": True,
            "created_at": utc_now(),
            "bundle_id": bundle_id,
            "training_commit": TRAINING_COMMIT,
            "study_id": STUDY,
            "parent": {
                **metadata,
                "relative_path": "parent/latest.pt",
                "sha256": files["parent/latest.pt"]["sha256"],
            },
            "completion_processed_bases": completion_bases,
            "runtime": {
                "source": "parent/colab_run_manifest.json",
                "python": environment.get("python"),
                "platform": environment.get("platform"),
                "packages": runtime_packages,
                "cuda": environment.get("cuda"),
                "device_name": environment.get("device_name"),
            },
            "handoff_runtime": runtime_manifest(),
            "files": files,
        }
        atomic_json(bundle / "TRAILBLAZER_INPUT_MANIFEST.json", manifest)
        return manifest
    except BaseException:
        shutil.rmtree(bundle, ignore_errors=True)
        raise


def verify_input(bundle: Path, *, verify_all: bool = True) -> dict[str, Any]:
    manifest_path = bundle / "TRAILBLAZER_INPUT_MANIFEST.json"
    if not manifest_path.is_file():
        raise ContractError("TRAILBLAZER_INPUT_MANIFEST.json is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("training_commit") != TRAINING_COMMIT or manifest.get("study_id") != STUDY:
        raise ContractError("wrong training commit or study identity")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ContractError("input file inventory is invalid")
    required = {
        "dataset/token_stream_manifest.json", "panels/e25.json", "panels/e100_additions.json",
        "panels/validation.json", "qualification/qualification_selection.json",
        "parent/latest.pt", "parent/run_manifest.json", "parent/LIVE_STATUS.json",
        "parent/colab_run_manifest.json",
        "parent/scale_gate.json", "study/protocol.json", f"study/{AMENDMENT_NAME}",
    }
    if missing := required.difference(files):
        raise ContractError(f"trailblazer input is missing required files: {sorted(missing)}")
    selected = files.items() if verify_all else ((name, files[name]) for name in required)
    for relative, record in selected:
        path = bundle / relative
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ContractError(f"trailblazer input checksum mismatch: {relative}")
    if json.loads((bundle / "parent/scale_gate.json").read_text()).get("proceed") is not True:
        raise ContractError("E25 scale gate does not say proceed")
    run_manifest = json.loads((bundle / "parent/run_manifest.json").read_text())
    live = json.loads((bundle / "parent/LIVE_STATUS.json").read_text())
    if run_manifest.get("stop_reason") != "panel_exhausted" or not run_manifest.get("scheduler_exhausted"):
        raise ContractError("E25 parent did not exhaust its panel")
    if live.get("state") != "completed":
        raise ContractError("E25 LIVE_STATUS is not completed")
    metadata = checkpoint_metadata(bundle / "parent/latest.pt")
    for key in ("optimizer_step", "processed_bases", "dataset_fingerprint", "code_commit"):
        if metadata[key] != manifest["parent"].get(key):
            raise ContractError(f"E25 checkpoint identity mismatch for {key}")
    model = metadata["model_config"]
    expected = {
        "block_count": 12, "d_model": 256, "num_heads": 8, "persistent_tokens": 4,
        "memory_depth": 2, "memory_architecture": "paper_residual_mlp_v2",
        "memory_recurrence_policy": "paper_exact", "gradient_horizon": 3,
        "memory_mode": "adaptive",
    }
    if any(model.get(key) != value for key, value in expected.items()):
        raise ContractError("E25 checkpoint scientific configuration drifted")
    qualification = json.loads((bundle / "qualification/qualification_selection.json").read_text())
    if qualification.get("passed") is not True:
        raise ContractError("C18 A100 qualification did not pass")
    return manifest


def _train_executable(python_bin: Path) -> str:
    candidate = python_bin.parent / "seqtrainer-titans-stage-c-train"
    return str(candidate)


def training_command(
    bundle: Path, run_dir: Path, python_bin: Path, *, cap_bases: int | None = None,
    max_optimizer_steps: int | None = None, first_chunk: bool = False,
    final: bool = False,
) -> list[str]:
    manifest = verify_input(bundle, verify_all=False)
    q = json.loads((bundle / "qualification/qualification_selection.json").read_text())
    if final and (cap_bases is not None or max_optimizer_steps is not None):
        raise ContractError("the final E100 chunk must exhaust the panel without a stop cap")
    if cap_bases == E50_BASES:
        run_id = "medium_adaptive_e50_milestone_v1"
    elif cap_bases == E75_BASES:
        run_id = "medium_adaptive_e75_milestone_v1"
    elif final:
        run_id = E100_RUN_ID
    else:
        run_id = "medium_adaptive_curc_chunk_v1"
    command = [
        _train_executable(python_bin),
        "--dataset-dir", str(bundle / "dataset"),
        "--panel-manifest", str(bundle / "panels/e100_additions.json"),
        "--validation-panel-manifest", str(bundle / "panels/validation.json"),
        "--run-dir", str(run_dir),
        "--memory-mode", "adaptive", "--horizon", "3",
        "--batch-size", str(q["batch_size"]), "--seed", "20260751",
        "--scheduler-policy", "stateful_rotation", "--scheduler-burst-segments", "96",
        "--checkpoint-every", "250", "--learning-rate", "3e-5",
        "--min-learning-rate", "3e-6", "--lr-warmup-bases", "2000000",
        "--lr-decay-bases", "100000000", "--weight-decay", "0.1",
        "--gradient-clip-norm", "0.5", "--activation", str(q["activation"]),
        "--block-count", "12", "--d-model", "256", "--num-heads", "8",
        "--persistent-tokens", "4", "--validation-streams", "1",
        "--validation-segments", "1",
        *DEEP_FLAGS,
        "--protocol", str(bundle / "study/protocol.json"),
        "--protocol-amendment", str(bundle / f"study/{AMENDMENT_NAME}"),
        "--run-id", run_id,
    ]
    if first_chunk:
        command.extend(["--warm-start-checkpoint", str(bundle / manifest["parent"]["relative_path"]), "--no-resume"])
    if final:
        command.append("--require-panel-completion")
    elif cap_bases is not None:
        command.extend(["--max-valid-bases", str(cap_bases)])
    elif max_optimizer_steps is not None:
        command.extend(["--max-optimizer-steps", str(max_optimizer_steps)])
    else:
        raise ContractError("a non-final chunk requires a base or optimizer-step cap")
    return command


def execution_chunks(
    start_bases: int, completion_bases: int, bases_per_second: float,
    walltime_hours: int = DEFAULT_WALLTIME_HOURS,
) -> list[dict[str, Any]]:
    if not 0 < start_bases < completion_bases or bases_per_second <= 0:
        raise ContractError("invalid continuation range or throughput")
    # Reserve 15% of each seven-day allocation for startup, checkpoints, and queue signals.
    safe_increment = max(1, int(bases_per_second * walltime_hours * 3600 * 0.85))
    cursor = start_bases
    boundaries = [value for value in (E50_BASES, E75_BASES) if cursor < value < completion_bases]
    boundaries.append(completion_bases)
    chunks: list[dict[str, Any]] = []
    for boundary in boundaries:
        while boundary - cursor > safe_increment:
            cursor += safe_increment
            chunks.append({"cap_bases": cursor, "milestone": None, "final": False})
        cursor = boundary
        label = "e50" if boundary == E50_BASES else "e75" if boundary == E75_BASES else "e100"
        chunks.append({
            "cap_bases": None if boundary == completion_bases else boundary,
            "milestone": label,
            "final": boundary == completion_bases,
        })
    for index, chunk in enumerate(chunks):
        chunk["index"] = index
        chunk["first_chunk"] = index == 0
    return chunks


def _container_prefix(image: Path | None) -> list[str]:
    return [] if image is None else ["apptainer", "exec", "--nv", str(image)]


def run_chunk(args: argparse.Namespace) -> dict[str, Any]:
    verify_input(args.bundle)
    latest = args.run_dir / "latest.pt"
    first = not latest.exists()
    command = training_command(
        args.bundle, args.run_dir, args.python_bin, cap_bases=args.cap_bases,
        max_optimizer_steps=args.max_optimizer_steps, first_chunk=first, final=args.final,
    )
    run_command([*_container_prefix(args.image), *command], cwd=args.repo)
    metadata = checkpoint_metadata(latest)
    if args.cap_bases is not None and metadata["processed_bases"] < args.cap_bases:
        raise ContractError("chunk exited below its processed-base boundary")
    if args.final:
        completed = json.loads((args.run_dir / "run_manifest.json").read_text())
        if not completed.get("scheduler_exhausted") or completed.get("stop_reason") != "panel_exhausted":
            raise ContractError("final E100 chunk did not exhaust the additions panel")
    return {"command": command, "checkpoint": metadata, "final": args.final}


def _verify_adaptive_checkpoint(path: Path, scientific_seed: int) -> dict[str, Any]:
    metadata = checkpoint_metadata(path)
    if metadata["code_commit"] != TRAINING_COMMIT:
        raise ContractError("adaptive checkpoint came from the wrong training commit")
    if metadata["model_config"].get("memory_mode") != "adaptive":
        raise ContractError("checkpoint is not an adaptive-memory trajectory")
    if metadata.get("scheduler_seed") != scientific_seed:
        raise ContractError("checkpoint scheduler seed differs from the declared trajectory")
    return metadata


def package_milestone(
    run_dir: Path, output: Path, milestone: str, *,
    trajectory_id: str = "adaptive-seed-20260751",
    execution_role: str = "curc-primary", scientific_seed: int = PRIMARY_SEED,
) -> dict[str, Any]:
    if milestone not in {"e25", "e50", "e75", "e100"}:
        raise ContractError("milestone must be e25, e50, e75, or e100")
    source = run_dir / "latest.pt"
    if not source.is_file():
        raise ContractError("authoritative CURC checkpoint is missing")
    metadata = _verify_adaptive_checkpoint(source, scientific_seed)
    threshold = {"e25": 1, "e50": E50_BASES, "e75": E75_BASES, "e100": 0}[milestone]
    if metadata["processed_bases"] < threshold:
        raise ContractError(f"checkpoint has not reached {milestone}")
    if milestone in {"e25", "e100"}:
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise ContractError(f"{milestone.upper()} run manifest is missing")
        completed = json.loads(manifest_path.read_text())
        if not completed.get("scheduler_exhausted") or completed.get("stop_reason") != "panel_exhausted":
            raise ContractError(f"{milestone.upper()} run is incomplete")
    digest = sha256_file(source)
    bundle_id = f"{milestone}-step-{metadata['optimizer_step']:09d}-{digest[:16]}"
    destination = output / bundle_id
    if destination.exists():
        raise ContractError(f"immutable milestone bundle already exists: {destination}")
    files: dict[str, Any] = {}
    selected = [source]
    for name in (
        "run_manifest.json", "LIVE_STATUS.json", "RUN_START.json", "training_history.json",
        "MODEL_ARCHITECTURE.txt", "architecture.json", "resume_verification.json",
    ):
        candidate = run_dir / name
        if candidate.is_file():
            selected.append(candidate)
    logs = run_dir / "logs"
    if logs.is_dir():
        selected.extend(iter_files(logs))
    for item in selected:
        relative = Path("latest.pt") if item == source else item.relative_to(run_dir)
        files[relative.as_posix()] = stable_copy(item, destination / relative)
    files["model.pt"] = stable_copy(source, destination / "model.pt")
    manifest = {
        "format_version": 1, "immutable": True, "created_at": utc_now(),
        "milestone": milestone, "bundle_id": bundle_id, "training_commit": TRAINING_COMMIT,
        "trajectory_id": trajectory_id, "execution_role": execution_role,
        "scientific_seed": scientific_seed,
        "checkpoint": {**metadata, "relative_path": "latest.pt", "sha256": digest},
        "evaluation_model": {"relative_path": "model.pt", "sha256": digest},
        "files": files,
    }
    atomic_json(destination / "MILESTONE_MANIFEST.json", manifest)
    atomic_json(destination / "MODEL_POINTER.json", {
        "format_version": 1, "milestone": milestone, "checkpoint": "latest.pt",
        "model_path": "model.pt", "trajectory_id": trajectory_id,
        "execution_role": execution_role, "scientific_seed": scientific_seed,
        "sha256": digest, "optimizer_step": metadata["optimizer_step"],
        "processed_bases": metadata["processed_bases"],
    })
    return manifest


def verify_milestone(bundle: Path) -> dict[str, Any]:
    path = bundle / "MILESTONE_MANIFEST.json"
    if not path.is_file():
        raise ContractError("MILESTONE_MANIFEST.json is missing")
    manifest = json.loads(path.read_text())
    if manifest.get("training_commit") != TRAINING_COMMIT or manifest.get("milestone") not in {"e25", "e50", "e75", "e100"}:
        raise ContractError("invalid milestone identity")
    for relative, record in manifest.get("files", {}).items():
        target = bundle / relative
        if not target.is_file() or sha256_file(target) != record.get("sha256"):
            raise ContractError(f"milestone checksum mismatch: {relative}")
    metadata = checkpoint_metadata(bundle / manifest["checkpoint"]["relative_path"])
    for key in ("optimizer_step", "processed_bases", "code_commit"):
        if metadata[key] != manifest["checkpoint"].get(key):
            raise ContractError(f"milestone checkpoint identity mismatch for {key}")
    model = manifest.get("evaluation_model", {})
    model_path = bundle / str(model.get("relative_path", ""))
    if not model_path.is_file() or sha256_file(model_path) != manifest["checkpoint"].get("sha256"):
        raise ContractError("milestone model.pt identity does not match its checkpoint")
    return manifest


def package_recovery(
    run_dir: Path, output: Path, *, execution_role: str = "curc-primary",
    scientific_seed: int = PRIMARY_SEED,
) -> dict[str, Any]:
    """Freeze the newest exact-resume state for laptop-mediated GCP failover."""
    source = run_dir / "latest.pt"
    if not source.is_file():
        raise ContractError("authoritative recovery checkpoint is missing")
    metadata = _verify_adaptive_checkpoint(source, scientific_seed)
    digest = sha256_file(source)
    trajectory_id = f"adaptive-seed-{scientific_seed}"
    bundle_id = f"recovery-{trajectory_id}-step-{metadata['optimizer_step']:09d}-{digest[:16]}"
    destination = output / bundle_id
    if destination.exists():
        raise ContractError(f"immutable recovery bundle already exists: {destination}")
    selected = [source]
    for name in (
        "run_manifest.json", "LIVE_STATUS.json", "RUN_START.json",
        "training_history.json", "MODEL_ARCHITECTURE.txt", "architecture.json",
        "resume_verification.json",
    ):
        candidate = run_dir / name
        if candidate.is_file():
            selected.append(candidate)
    files: dict[str, Any] = {}
    try:
        for item in selected:
            relative = Path("latest.pt") if item == source else item.relative_to(run_dir)
            files[relative.as_posix()] = stable_copy(item, destination / relative)
        manifest = {
            "format_version": 1, "immutable": True, "created_at": utc_now(),
            "bundle_id": bundle_id, "training_commit": TRAINING_COMMIT,
            "study_id": STUDY, "trajectory_id": trajectory_id,
            "execution_role": execution_role, "scientific_seed": scientific_seed,
            "checkpoint": {**metadata, "relative_path": "latest.pt", "sha256": digest},
            "files": files,
        }
        atomic_json(destination / "RECOVERY_MANIFEST.json", manifest)
        return manifest
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def verify_recovery(bundle: Path) -> dict[str, Any]:
    manifest_path = bundle / "RECOVERY_MANIFEST.json"
    if not manifest_path.is_file():
        raise ContractError("RECOVERY_MANIFEST.json is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("training_commit") != TRAINING_COMMIT or manifest.get("study_id") != STUDY:
        raise ContractError("invalid recovery study or training commit")
    if manifest.get("execution_role") != "curc-primary":
        raise ContractError("GCP failover accepts only a CURC-primary recovery bundle")
    seed = int(manifest.get("scientific_seed", -1))
    if seed != PRIMARY_SEED or manifest.get("trajectory_id") != f"adaptive-seed-{PRIMARY_SEED}":
        raise ContractError("recovery is not the primary adaptive trajectory")
    for relative, record in manifest.get("files", {}).items():
        target = bundle / relative
        if not target.is_file() or sha256_file(target) != record.get("sha256"):
            raise ContractError(f"recovery checksum mismatch: {relative}")
    metadata = _verify_adaptive_checkpoint(bundle / manifest["checkpoint"]["relative_path"], seed)
    for key in ("optimizer_step", "processed_bases", "dataset_fingerprint", "code_commit"):
        if metadata[key] != manifest["checkpoint"].get(key):
            raise ContractError(f"recovery checkpoint identity mismatch for {key}")
    return manifest


def allocation_brief(output: Path) -> dict[str, Any]:
    estimated_hours = 450
    estimated_su = math.ceil(estimated_hours * (GPU_BILLING_WEIGHT + DEFAULT_CPU_COUNT * CPU_BILLING_WEIGHT))
    payload = {
        "title": "Stage C adaptive-memory E. coli language-model trailblazer",
        "field": "Computational biology and machine learning",
        "partition": "aa100", "qos": "gpu-long", "gpu": "a100-40gb",
        "gpus_per_job": 1, "cpus_per_job": DEFAULT_CPU_COUNT, "memory_gb": 64,
        "estimated_a100_hours": estimated_hours, "estimated_service_units": estimated_su,
        "checkpointing": "Atomic full-state checkpoints every 250 optimizer steps; exact scheduler, optimizer, RNG, and functional-memory resume across dependent jobs.",
        "software": "Apptainer, CUDA, PyTorch, and SeqTrainer pinned to commit " + TRAINING_COMMIT,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CURC Ascent allocation brief", "", f"**Title:** {payload['title']}", "",
        "This project continues one preregistered 25M-parameter adaptive-memory genomic language model from E25 to E100 on one NVIDIA A100 40 GB GPU.", "",
        f"- Requested resource: `{payload['partition']}`, `{payload['qos']}`, one `{payload['gpu']}`",
        f"- Estimate: {estimated_hours} A100-hours; approximately {estimated_su:,} SUs including {DEFAULT_CPU_COUNT} CPU cores",
        f"- Checkpointing: {payload['checkpointing']}", f"- Software: {payload['software']}",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(output.with_suffix(".json"), payload)
    return payload


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    manifest = verify_input(args.bundle, verify_all=not args.fast)
    for executable in ("apptainer", "sbatch", "squeue"):
        if shutil.which(executable) is None:
            raise ContractError(f"required CURC executable is unavailable: {executable}")
    if args.image and not args.image.is_file():
        raise ContractError("Apptainer image is missing")
    input_bytes = sum(int(item["bytes"]) for item in manifest["files"].values())
    free = shutil.disk_usage(args.work_root).free
    required = max(20 * 1024**3, int(input_bytes * 1.5))
    if free < required:
        raise ContractError("CURC work filesystem has insufficient free capacity")
    report = {
        "format_version": 1, "checked_at": utc_now(), "bundle_id": manifest["bundle_id"],
        "input_bytes": input_bytes, "free_bytes": free, "required_free_bytes": required,
        "partition": "aa100", "qos": "gpu-long", "gpu": "a100-40gb",
    }
    if args.on_compute:
        probe = run_command([
            *_container_prefix(args.image), str(args.python_bin), "-c",
            "import json,torch; assert torch.cuda.is_available(); n=torch.cuda.get_device_name(0); assert 'A100' in n.upper(); print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':n}))",
        ], capture=True)
        report["runtime"] = json.loads(probe.stdout)
        expected = manifest.get("runtime", {})
        expected_packages = expected.get("packages", {}) if isinstance(expected, Mapping) else {}
        expected_torch = expected_packages.get("torch") if isinstance(expected_packages, Mapping) else None
        if expected_torch and report["runtime"]["torch"] != expected_torch:
            raise ContractError(
                f"CURC torch {report['runtime']['torch']} differs from E25 runtime {expected_torch}"
            )
    return report


def prepare_runtime(args: argparse.Namespace) -> dict[str, Any]:
    manifest = verify_input(args.bundle, verify_all=False)
    if not args.image.is_file():
        raise ContractError("Apptainer image is missing")
    args.venv.parent.mkdir(parents=True, exist_ok=True)
    if not (args.venv / "bin/python").is_file():
        run_command(["apptainer", "exec", "--nv", str(args.image), "python3", "-m", "venv",
                     "--system-site-packages", str(args.venv)])
    python_bin = args.venv / "bin/python"
    run_command(["apptainer", "exec", "--nv", str(args.image), str(python_bin), "-m", "pip",
                 "install", "--no-deps", "-e", str(args.repo)])
    probe = run_command([
        "apptainer", "exec", "--nv", str(args.image), str(python_bin), "-c",
        "import json,torch,numpy,pandas,pyarrow; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':numpy.__version__,'pandas':pandas.__version__,'pyarrow':pyarrow.__version__}))",
    ], capture=True)
    actual = json.loads(probe.stdout)
    expected = manifest.get("runtime", {}).get("packages", {})
    mismatches = {
        name: {"expected": expected[name], "actual": actual.get(name)}
        for name in ("torch", "numpy", "pandas", "pyarrow")
        if expected.get(name) and actual.get(name) != expected[name]
    }
    if mismatches:
        raise ContractError(f"Apptainer runtime does not recreate the E25 manifest: {mismatches}")
    return {"python_bin": str(python_bin), "image": str(args.image), "versions": actual}


def pilot(args: argparse.Namespace) -> dict[str, Any]:
    manifest = verify_input(args.bundle)
    args.output.mkdir(parents=True, exist_ok=True)
    verify_output = args.output / "resume_verification.json"
    resume_command = [
        str(args.python_bin.parent / "seqtrainer-titans-stage-c-resume-verify"),
        "--dataset-dir", str(args.bundle / "dataset"),
        "--panel-manifest", str(args.bundle / "panels/e25.json"),
        "--checkpoint", str(args.bundle / "parent/latest.pt"),
        "--output", str(verify_output), "--device", "cuda",
        "--expected-step", str(manifest["parent"]["optimizer_step"]),
    ]
    run_command([*_container_prefix(args.image), *resume_command], cwd=args.repo)
    pilot_run = args.output / "one_step"
    start = time.monotonic()
    command = training_command(
        args.bundle, pilot_run, args.python_bin,
        max_optimizer_steps=int(manifest["parent"]["optimizer_step"]) + 1,
        first_chunk=True,
    )
    run_command([*_container_prefix(args.image), *command], cwd=args.repo)
    elapsed = time.monotonic() - start
    metadata = checkpoint_metadata(pilot_run / "latest.pt")
    delta = metadata["processed_bases"] - int(manifest["parent"]["processed_bases"])
    if metadata["optimizer_step"] != int(manifest["parent"]["optimizer_step"]) + 1 or delta <= 0:
        raise ContractError("CURC pilot did not perform exactly one advancing optimizer step")
    report = {
        "format_version": 1, "completed_at": utc_now(), "gpu": "A100 40 GB",
        "starting_optimizer_step": manifest["parent"]["optimizer_step"],
        "continued_optimizer_step": metadata["optimizer_step"],
        "processed_base_delta": delta, "elapsed_seconds": elapsed,
        "bases_per_second": delta / elapsed,
        "resume_verification": str(verify_output),
    }
    atomic_json(args.output / "CURC_PILOT_REPORT.json", report)
    return report


def _slurm_script(
    *, account: str, launcher: Path, bundle: Path, run_dir: Path, output: Path,
    repo: Path, python_bin: Path, image: Path | None, chunk: Mapping[str, Any],
    walltime_hours: int,
) -> str:
    days, hours = divmod(walltime_hours, 24)
    command = [
        sys.executable, str(launcher), "run-chunk", "--bundle", str(bundle),
        "--run-dir", str(run_dir), "--repo", str(repo), "--python-bin", str(python_bin),
    ]
    if image:
        command.extend(["--image", str(image)])
    if chunk["final"]:
        command.append("--final")
    else:
        command.extend(["--cap-bases", str(chunk["cap_bases"])])
    quoted = " ".join(shlex.quote(part) for part in command)
    package = ""
    if chunk.get("milestone"):
        package_command = [
            sys.executable, str(launcher), "package", "--run-dir", str(run_dir),
            "--output", str(output), "--milestone", str(chunk["milestone"]),
        ]
        package = "\n" + " ".join(shlex.quote(part) for part in package_command)
    return f"""#!/bin/bash
#SBATCH --account={account}
#SBATCH --partition=aa100
#SBATCH --qos=gpu-long
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={DEFAULT_CPU_COUNT}
#SBATCH --mem=64G
#SBATCH --time={days}-{hours:02d}:00:00
#SBATCH --job-name=c20-{chunk['index']:02d}
#SBATCH --output={output}/slurm-%x-%j.log
#SBATCH --mail-type=FAIL,END
set -euo pipefail
{quoted}{package}
"""


def submit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = verify_input(args.bundle)
    pilot_report = json.loads(args.pilot_report.read_text())
    throughput = float(pilot_report.get("bases_per_second", 0))
    chunks = execution_chunks(
        int(manifest["parent"]["processed_bases"]),
        int(manifest["completion_processed_bases"]), throughput,
        args.walltime_hours,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    scripts = args.output / "slurm_scripts"
    scripts.mkdir(exist_ok=True)
    jobs: list[dict[str, Any]] = []
    dependency: str | None = None
    for chunk in chunks:
        path = scripts / f"chunk-{chunk['index']:02d}.sbatch"
        path.write_text(_slurm_script(
            account=args.account, launcher=Path(__file__).resolve(), bundle=args.bundle,
            run_dir=args.run_dir, output=args.output, repo=args.repo,
            python_bin=args.python_bin, image=args.image, chunk=chunk,
            walltime_hours=args.walltime_hours,
        ), encoding="utf-8")
        record = {**chunk, "script": str(path), "job_id": None}
        if not args.dry_run:
            command = ["sbatch", "--parsable"]
            if dependency:
                command.extend(["--dependency", f"afterok:{dependency}"])
            command.append(str(path))
            result = run_command(command, capture=True)
            dependency = result.stdout.strip().split(";", 1)[0]
            record["job_id"] = dependency
        jobs.append(record)
    plan = {
        "format_version": 1, "created_at": utc_now(), "account": args.account,
        "bundle_id": manifest["bundle_id"], "bases_per_second": throughput,
        "walltime_hours": args.walltime_hours, "jobs": jobs,
        "estimated_a100_hours": (
            int(manifest["completion_processed_bases"]) - int(manifest["parent"]["processed_bases"])
        ) / throughput / 3600,
    }
    atomic_json(args.output / "CURC_EXECUTION_PLAN.json", plan)
    return plan


def status(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {"checked_at": utc_now(), "run_dir": str(args.run_dir)}
    latest = args.run_dir / "latest.pt"
    if latest.is_file():
        report["checkpoint"] = {**checkpoint_metadata(latest), "sha256": sha256_file(latest)}
    live = args.run_dir / "LIVE_STATUS.json"
    if live.is_file():
        report["live"] = json.loads(live.read_text())
    plan = args.output / "CURC_EXECUTION_PLAN.json"
    if plan.is_file():
        report["execution_plan"] = json.loads(plan.read_text())
        job_ids = [str(item["job_id"]) for item in report["execution_plan"]["jobs"] if item.get("job_id")]
        if job_ids and shutil.which("squeue"):
            queue = run_command(["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i|%T|%M|%R"], capture=True, check=False)
            report["slurm"] = queue.stdout.splitlines()
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="create the post-gate E25-to-CURC input bundle")
    export.add_argument("--drive-root", type=Path, required=True)
    export.add_argument("--repo-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    brief = sub.add_parser("allocation-brief", help="write a CURC Ascent application brief")
    brief.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare-runtime", help="create and verify the pinned Apptainer venv")
    prepare.add_argument("--bundle", type=Path, required=True)
    prepare.add_argument("--repo", type=Path, required=True)
    prepare.add_argument("--image", type=Path, required=True)
    prepare.add_argument("--venv", type=Path, required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--bundle", type=Path, required=True)
    common.add_argument("--repo", type=Path, required=True)
    common.add_argument("--python-bin", type=Path, required=True)
    common.add_argument("--image", type=Path)
    pre = sub.add_parser("preflight", parents=[common], help="verify CURC input, tools, storage, and optional GPU")
    pre.add_argument("--work-root", type=Path, required=True)
    pre.add_argument("--fast", action="store_true")
    pre.add_argument("--on-compute", action="store_true")
    pilot_cmd = sub.add_parser("pilot", parents=[common], help="verify exact resume and measure one A100 step")
    pilot_cmd.add_argument("--output", type=Path, required=True)
    chunk = sub.add_parser("run-chunk", parents=[common], help="execute one bounded or final continuation chunk")
    chunk.add_argument("--run-dir", type=Path, required=True)
    chunk.add_argument("--cap-bases", type=int)
    chunk.add_argument("--max-optimizer-steps", type=int)
    chunk.add_argument("--final", action="store_true")
    submit_cmd = sub.add_parser("submit", parents=[common], help="submit dependency-linked GPU-long jobs")
    submit_cmd.add_argument("--account", required=True)
    submit_cmd.add_argument("--pilot-report", type=Path, required=True)
    submit_cmd.add_argument("--run-dir", type=Path, required=True)
    submit_cmd.add_argument("--output", type=Path, required=True)
    submit_cmd.add_argument("--walltime-hours", type=int, default=DEFAULT_WALLTIME_HOURS)
    submit_cmd.add_argument("--dry-run", action="store_true")
    package = sub.add_parser("package", help="freeze an immutable CURC milestone bundle")
    package.add_argument("--run-dir", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--milestone", choices=("e25", "e50", "e75", "e100"), required=True)
    package.add_argument("--trajectory-id", default=f"adaptive-seed-{PRIMARY_SEED}")
    package.add_argument("--execution-role", default="curc-primary")
    package.add_argument("--scientific-seed", type=int, default=PRIMARY_SEED)
    verify = sub.add_parser("verify-milestone", help="verify a returned milestone bundle")
    verify.add_argument("--bundle", type=Path, required=True)
    recovery = sub.add_parser("export-recovery", help="freeze the latest CURC state for GCP failover")
    recovery.add_argument("--run-dir", type=Path, required=True)
    recovery.add_argument("--output", type=Path, required=True)
    recovery.add_argument("--execution-role", default="curc-primary")
    recovery.add_argument("--scientific-seed", type=int, default=PRIMARY_SEED)
    verify_recovery_cmd = sub.add_parser("verify-recovery", help="verify a cross-cloud recovery bundle")
    verify_recovery_cmd.add_argument("--bundle", type=Path, required=True)
    stat = sub.add_parser("status", help="show checkpoint and SLURM-chain status")
    stat.add_argument("--run-dir", type=Path, required=True)
    stat.add_argument("--output", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "export":
            result = export_input(args.drive_root, args.repo_root, args.output)
        elif args.command == "allocation-brief":
            result = allocation_brief(args.output)
        elif args.command == "prepare-runtime":
            result = prepare_runtime(args)
        elif args.command == "preflight":
            result = preflight(args)
        elif args.command == "pilot":
            result = pilot(args)
        elif args.command == "run-chunk":
            result = run_chunk(args)
        elif args.command == "submit":
            result = submit(args)
        elif args.command == "package":
            result = package_milestone(
                args.run_dir, args.output, args.milestone,
                trajectory_id=args.trajectory_id, execution_role=args.execution_role,
                scientific_seed=args.scientific_seed,
            )
        elif args.command == "verify-milestone":
            result = verify_milestone(args.bundle)
        elif args.command == "export-recovery":
            result = package_recovery(
                args.run_dir, args.output, execution_role=args.execution_role,
                scientific_seed=args.scientific_seed,
            )
        elif args.command == "verify-recovery":
            result = verify_recovery(args.bundle)
        elif args.command == "status":
            result = status(args)
        else:
            raise AssertionError(args.command)
    except (ContractError, FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
