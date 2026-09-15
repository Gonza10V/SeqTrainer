from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts/stage_c_c19_gcp.py"
SPEC = importlib.util.spec_from_file_location("stage_c_c19_gcp", SCRIPT)
assert SPEC and SPEC.loader
gcp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gcp)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def checkpoint(
    path: Path, step: int, bases: int = 1000,
    dataset_fingerprint: str = "dataset-panel-fingerprint",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "trainer_state": {"optimizer_step": step, "processed_bases": bases},
            "scheduler_state": {
                "policy": "stateful_rotation", "batch_size": 1,
                "burst_segments": 96, "seed": 20260751, "cursor": step,
            },
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_components": {"dataset": "abc", "train_panel": "e25"},
            "code_commit": gcp.TRAINING_COMMIT,
            "model_config": {
                "block_count": 12, "d_model": 256, "num_heads": 8, "persistent_tokens": 4,
                "memory_depth": 2, "memory_architecture": "paper_residual_mlp_v2",
                "memory_expansion_factor": 4, "memory_projection_convolution_kernel": 4,
                "memory_normalize_queries_and_keys": True,
                "memory_gate_granularity": "per_layer_channel",
                "memory_recurrence_policy": "paper_exact", "memory_surprise_clip_norm": None,
                "memory_associative_loss_reduction": "sum", "memory_max_gradient_rms": None,
                "memory_max_gradient_rms_ratio": None, "memory_theta_max": 1.0,
                "memory_alpha_initial": 0.001, "memory_eta_initial": 0.9,
                "memory_theta_initial": 0.001, "gradient_horizon": 3,
                "memory_mode": "adaptive", "backend": {"activation_dtype": "bfloat16"},
            },
        },
        path,
    )


def drive_fixture(tmp_path: Path, step: int = 17) -> Path:
    drive = tmp_path / "drive"
    dataset = drive / "stage_c_dataset/ordered_streams/nonoverlap_6mer_v1"
    index = dataset / "stream_index.jsonl"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_bytes(b"{}\n")
    tokens = dataset / "tokens_00000.npy"
    lengths = dataset / "base_lengths_00000.npy"
    tokens.write_bytes(b"ordered token bytes")
    lengths.write_bytes(b"base lengths")
    dataset_manifest = {
        "format_version": 1, "index": index.name, "index_sha256": gcp.sha256_file(index),
        "shards": [{
            "tokens": tokens.name, "tokens_sha256": gcp.sha256_file(tokens),
            "base_lengths": lengths.name, "base_lengths_sha256": gcp.sha256_file(lengths),
        }],
    }
    write_json(dataset / "token_stream_manifest.json", dataset_manifest)
    dataset_sha = gcp.sha256_file(dataset / "token_stream_manifest.json")
    panels = drive / "study/stage_c_ecoli_medium_deep_memory_v3/panels"
    train_panel = {
        "format_version": 1, "panel_id": "e25", "role": "train", "split": "train",
        "parent_dataset_fingerprint": dataset_sha, "stream_ids": ["train:1"],
        "accessions": ["GCF_1.1"], "predictable_bases": 10, "selection_order": [],
    }
    validation_panel = {
        "format_version": 1, "panel_id": "validation", "role": "validation", "split": "val",
        "parent_dataset_fingerprint": dataset_sha, "stream_ids": ["val:1"],
        "accessions": ["GCF_2.1"], "predictable_bases": 10, "selection_order": [],
    }
    write_json(panels / "e25.json", train_panel)
    write_json(panels / "validation.json", validation_panel)
    write_json(panels / "panel_summary.json", {"nested": True})
    study = drive / "study/stage_c_ecoli_medium_deep_memory_v3"
    write_json(study / "protocol.json", {"study_id": "stage_c_ecoli_medium_deep_memory_v3"})
    (study / "ledger.jsonl").write_text('{"event_hash":"prior"}\n', encoding="utf-8")
    write_json(
        drive / "runs/c18_v3_medium_a100_qualification/qualification_selection.json",
        {"passed": True, "activation": "bfloat16", "batch_size": 1},
    )
    run = drive / "runs" / gcp.RUN_NAME
    def panel_hash(value: object) -> str:
        return gcp.hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    fingerprint = gcp.hashlib.sha256(json.dumps({
        "dataset": dataset_sha,
        "train_panel": panel_hash(train_panel),
        "validation_panel": panel_hash(validation_panel),
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    checkpoint(run / "latest.pt", step, dataset_fingerprint=fingerprint)
    write_json(run / "LIVE_STATUS.json", {"state": "stopped", "optimizer_steps": step})
    write_json(run / "training_history.json", [{"optimizer_step": step + 1}])
    return drive


def exported_bundle(tmp_path: Path, step: int = 17) -> tuple[Path, dict]:
    drive = drive_fixture(tmp_path, step)
    output = tmp_path / "exports"
    manifest = gcp.export_cutover(drive, output)
    return output / manifest["cutover_id"], manifest


def test_training_command_preserves_frozen_03l_scientific_configuration(tmp_path: Path) -> None:
    bundle, _ = exported_bundle(tmp_path)
    command = gcp.training_command(bundle, tmp_path / "run")
    joined = " ".join(command)
    expected_pairs = {
        "--memory-mode": "adaptive", "--horizon": "3", "--batch-size": "1",
        "--seed": "20260751", "--scheduler-policy": "stateful_rotation",
        "--scheduler-burst-segments": "96", "--checkpoint-every": "250",
        "--learning-rate": "3e-5", "--min-learning-rate": "3e-6",
        "--lr-warmup-bases": "2000000", "--lr-decay-bases": "100000000",
        "--weight-decay": "0.1", "--gradient-clip-norm": "0.5",
        "--activation": "bfloat16", "--block-count": "12", "--d-model": "256",
        "--num-heads": "8", "--persistent-tokens": "4", "--memory-depth": "2",
        "--memory-architecture": "paper_residual_mlp_v2",
        "--memory-recurrence-policy": "paper_exact",
        "--memory-associative-loss-reduction": "sum", "--memory-theta-max": "1.0",
        "--run-id": gcp.RUN_ID,
    }
    for flag, value in expected_pairs.items():
        index = command.index(flag)
        assert command[index + 1] == value
    assert "--require-panel-completion" in command
    assert "--memory-normalize-queries-and-keys" in command
    assert "--no-resume" not in command
    assert "--warm-start-checkpoint" not in command
    assert "medium_adaptive_e25_v1" in joined


def test_export_rejects_active_training_and_never_creates_bundle(tmp_path: Path) -> None:
    drive = drive_fixture(tmp_path)
    write_json(drive / "runs" / gcp.RUN_NAME / "LIVE_STATUS.json", {"state": "running"})
    with pytest.raises(gcp.ContractError, match="stop the Colab"):
        gcp.export_cutover(drive, tmp_path / "exports")
    assert not (drive / "runs" / gcp.RUN_NAME / "GCP_CUTOVER_LOCK.json").exists()


def test_stable_copy_rejects_source_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"before")
    original = shutil.copy2

    def mutating_copy(src: Path, dst: Path):
        result = original(src, dst)
        Path(src).write_bytes(b"after")
        return result

    monkeypatch.setattr(gcp.shutil, "copy2", mutating_copy)
    with pytest.raises(gcp.ContractError, match="changed during cutover"):
        gcp.stable_copy(source, tmp_path / "destination")
    assert not (tmp_path / "destination").exists()


def test_verify_bundle_rejects_checksum_and_wrong_commit(tmp_path: Path) -> None:
    bundle, _ = exported_bundle(tmp_path)
    (bundle / "panels/e25.json").write_text("tampered")
    with pytest.raises(gcp.ContractError, match="checksum mismatch"):
        gcp.verify_bundle(bundle)


def test_verify_bundle_rejects_wrong_panel_even_with_updated_inventory(tmp_path: Path) -> None:
    bundle, _ = exported_bundle(tmp_path)
    panel_path = bundle / "panels/e25.json"
    panel = json.loads(panel_path.read_text())
    panel["role"] = "validation"
    write_json(panel_path, panel)
    manifest_path = bundle / "CUTOVER_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["panels/e25.json"]["sha256"] = gcp.sha256_file(panel_path)
    write_json(manifest_path, manifest)
    with pytest.raises(gcp.ContractError, match="wrong e25 panel"):
        gcp.verify_bundle(bundle)
    bundle, _ = exported_bundle(tmp_path / "second")
    manifest_path = bundle / "CUTOVER_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training_commit"] = "wrong"
    write_json(manifest_path, manifest)
    with pytest.raises(gcp.ContractError, match="wrong training commit"):
        gcp.verify_bundle(bundle)


def test_newer_persistent_checkpoint_is_authoritative(tmp_path: Path) -> None:
    bundle, manifest = exported_bundle(tmp_path, step=20)
    work = tmp_path / "work"
    shutil.copytree(bundle, work / "cutover")
    checkpoint(work / "authoritative/runs" / gcp.RUN_NAME / "latest.pt", 21)
    config = {"bucket": "unused", "cutover_id": manifest["cutover_id"]}
    _, staged = gcp.stage_cutover(config, work)
    assert staged["checkpoint"]["optimizer_step"] == 20
    assert gcp.newest_checkpoint_step(work / "authoritative/runs" / gcp.RUN_NAME) == 21


def test_older_persistent_checkpoint_is_rejected(tmp_path: Path) -> None:
    bundle, manifest = exported_bundle(tmp_path, step=20)
    work = tmp_path / "work"
    shutil.copytree(bundle, work / "cutover")
    checkpoint(work / "authoritative/runs" / gcp.RUN_NAME / "latest.pt", 19)
    with pytest.raises(gcp.ContractError, match="regressed"):
        gcp.stage_cutover({"bucket": "unused", "cutover_id": manifest["cutover_id"]}, work)


def test_immutable_checkpoint_name_and_collision_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    digest = "a" * 64
    assert gcp.immutable_checkpoint_key(42, digest) == f"checkpoints/step-000000042-{digest}/latest.pt"
    source = tmp_path / "latest.pt"
    source.write_bytes(b"checkpoint")

    def collision(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, "", "conditionNotMet")

    monkeypatch.setattr(gcp, "run_command", collision)
    with pytest.raises(gcp.ContractError, match="collision"):
        gcp.upload_immutable(source, "gs://bucket/existing")


def test_checkpoint_mirroring_rejects_unexplained_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    checkpoint(run / "latest.pt", 42)
    monkeypatch.setattr(gcp, "upload_immutable", lambda source, uri: None)
    monkeypatch.setattr(
        gcp, "run_command",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )
    config = {"bucket": "bucket", "prefix": "prefix"}
    manifest = {"checkpoint": {"optimizer_step": 40}}
    gcp.mirror_outputs(run, config, manifest)
    checkpoint(run / "latest.pt", 41)
    with pytest.raises(gcp.ContractError, match="regression"):
        gcp.mirror_outputs(run, config, manifest)


def completed_download(tmp_path: Path, cutover_step: int, final_step: int) -> Path:
    run = tmp_path / "completed"
    checkpoint(run / "latest.pt", final_step)
    write_json(run / "run_manifest.json", {"scheduler_exhausted": True, "stop_reason": "panel_exhausted"})
    write_json(run / "LIVE_STATUS.json", {"state": "completed"})
    write_json(run / "MODEL_ARCHITECTURE.json", {"status": "passed"})
    write_json(run / "resume_verification.json", {"status": "passed"})
    write_json(run / "COMPLETE.json", {
        "architecture_passed": True, "resume_verification_passed": True,
        "cutover_step": cutover_step, "final_step": final_step,
        "checkpoint_sha256": gcp.sha256_file(run / "latest.pt"),
    })
    study_return = run / "GCP_STUDY_RETURN"
    (study_return / "ledger.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (study_return / "ledger.jsonl").write_text('{"event_hash":"c19"}\n', encoding="utf-8")
    write_json(study_return / "record_markers" / f"{gcp.RUN_ID}.json", {"run_id": gcp.RUN_ID})
    return run


def test_guarded_repatriation_publishes_only_completed_advanced_run(tmp_path: Path) -> None:
    drive = drive_fixture(tmp_path, step=10)
    cutover_checkpoint = drive / "runs" / gcp.RUN_NAME / "latest.pt"
    manifest = {
        "cutover_id": "step-000000010-test",
        "checkpoint": {"optimizer_step": 10, "sha256": gcp.sha256_file(cutover_checkpoint)},
        "files": {"study/ledger.jsonl": {"sha256": gcp.sha256_file(
            drive / "study/stage_c_ecoli_medium_deep_memory_v3/ledger.jsonl"
        )}},
    }
    downloaded = completed_download(tmp_path, 10, 11)
    gcp.atomic_repatriate(downloaded, drive, manifest)
    destination = drive / "runs" / gcp.RUN_NAME
    assert gcp.newest_checkpoint_step(destination) == 11
    assert (destination / "pre_gcp_cutover_archive" / manifest["cutover_id"] / "latest.pt").is_file()
    assert (drive / "study/stage_c_ecoli_medium_deep_memory_v3/record_markers" / f"{gcp.RUN_ID}.json").is_file()
    assert not list(destination.rglob("*.partial"))


@pytest.mark.parametrize(
    "mode", ["incomplete", "not_advanced", "hash_mismatch", "drive_advanced", "ledger_advanced"],
)
def test_guarded_repatriation_rejects_invalid_return(tmp_path: Path, mode: str) -> None:
    drive = drive_fixture(tmp_path, step=10)
    original = drive / "runs" / gcp.RUN_NAME / "latest.pt"
    manifest = {
        "cutover_id": "cutover",
        "checkpoint": {"optimizer_step": 10, "sha256": gcp.sha256_file(original)},
        "files": {"study/ledger.jsonl": {"sha256": gcp.sha256_file(
            drive / "study/stage_c_ecoli_medium_deep_memory_v3/ledger.jsonl"
        )}},
    }
    downloaded = completed_download(tmp_path, 10, 11)
    if mode == "incomplete":
        write_json(downloaded / "LIVE_STATUS.json", {"state": "failed"})
    elif mode == "not_advanced":
        downloaded = completed_download(tmp_path / "other", 10, 10)
    elif mode == "hash_mismatch":
        complete = json.loads((downloaded / "COMPLETE.json").read_text())
        complete["checkpoint_sha256"] = "0" * 64
        write_json(downloaded / "COMPLETE.json", complete)
    elif mode == "drive_advanced":
        checkpoint(original, 12)
    else:
        (drive / "study/stage_c_ecoli_medium_deep_memory_v3/ledger.jsonl").write_text(
            '{"event_hash":"independent"}\n', encoding="utf-8"
        )
    with pytest.raises(gcp.ContractError):
        gcp.atomic_repatriate(downloaded, drive, manifest)


def test_disk_sizing_is_at_least_200gb_and_150_percent() -> None:
    gib = 1024 ** 3
    assert gcp.choose_disk_size(1 * gib) == 200
    assert gcp.choose_disk_size(200 * gib) == 300


def test_service_restart_and_runtime_contract() -> None:
    assert "WantedBy=multi-user.target" in gcp.SYSTEMD_UNIT
    assert "Restart=no" in gcp.SYSTEMD_UNIT
    assert "TimeoutStartSec=infinity" in gcp.SYSTEMD_UNIT
    flags = gcp.vm_runtime_flags()
    assert flags[flags.index("--max-run-duration") + 1] == "36h"
    assert flags[flags.index("--instance-termination-action") + 1] == "STOP"
    assert flags[flags.index("--maintenance-policy") + 1] == "TERMINATE"
