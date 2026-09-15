from __future__ import annotations

import importlib.util
import ast
import json
from pathlib import Path
import shutil

import pytest
import torch

from seqtrainer.torch.titans_paper_mac_stage_c.study import StudyProtocol


ROOT = Path(__file__).parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


curc = load_script("stage_c_c20_curc_test", ROOT / "scripts/stage_c_c20_curc.py")
control = load_script("stage_c_e25_control_gcp_test", ROOT / "scripts/stage_c_e25_control_gcp.py")
adaptive_gcp = load_script(
    "stage_c_adaptive_e100_gcp_test", ROOT / "scripts/stage_c_adaptive_e100_gcp.py",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_checkpoint(
    path: Path, *, step: int, bases: int, mode: str = "adaptive",
    seed: int = curc.PRIMARY_SEED,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 2,
        "trainer_state": {
            "optimizer_step": step, "processed_bases": bases,
            "processed_segments": step * 3,
        },
        "scheduler_state": {
            "policy": "stateful_rotation", "batch_size": 1,
            "burst_segments": 96, "seed": seed,
        },
        "dataset_fingerprint": "e25-dataset-panel-fingerprint",
        "dataset_components": {"dataset": "dataset", "train_panel": "e25"},
        "code_commit": curc.TRAINING_COMMIT,
        "model_config": {
            "block_count": 12, "d_model": 256, "num_heads": 8,
            "persistent_tokens": 4, "memory_depth": 2,
            "memory_architecture": "paper_residual_mlp_v2",
            "memory_recurrence_policy": "paper_exact", "gradient_horizon": 3,
            "memory_mode": mode, "backend": {"activation_dtype": "bfloat16"},
        },
    }, path)


def drive_fixture(tmp_path: Path, *, gate: bool = True) -> Path:
    drive = tmp_path / "drive"
    dataset = drive / "stage_c_dataset/ordered_streams/nonoverlap_6mer_v1"
    write_json(dataset / "token_stream_manifest.json", {"format_version": 1})
    (dataset / "tokens.npy").write_bytes(b"tokens")
    study = drive / "study" / curc.STUDY
    panels = study / "panels"
    write_json(panels / "e25.json", {
        "panel_id": "e25", "role": "train", "split": "train",
        "predictable_bases": 26_062_903,
    })
    write_json(panels / "e100_additions.json", {
        "panel_id": "e100-additions", "role": "train", "split": "train",
        "predictable_bases": 75_000_000,
    })
    write_json(panels / "validation.json", {
        "panel_id": "validation", "role": "validation", "split": "val",
        "predictable_bases": 1_000,
    })
    write_json(panels / "test.json", {
        "panel_id": "test", "role": "test", "split": "test",
        "predictable_bases": 1_000,
    })
    shutil.copy2(ROOT / f"studies/{curc.STUDY}/protocol.json", study / "protocol.json")
    write_json(
        drive / "runs/c18_v3_medium_a100_qualification/qualification_selection.json",
        {"passed": True, "activation": "bfloat16", "batch_size": 1},
    )
    run = drive / "runs" / curc.E25_RUN
    write_checkpoint(run / "latest.pt", step=40_000, bases=26_062_903)
    write_json(run / "run_manifest.json", {
        "scheduler_exhausted": True, "stop_reason": "panel_exhausted",
        "optimizer_steps": 40_000, "processed_bases": 26_062_903,
    })
    write_json(run / "LIVE_STATUS.json", {"state": "completed"})
    write_json(run / "colab_run_manifest.json", {
        "format_version": 1,
        "environment": {
            "python": "3.12.13", "platform": "Linux", "torch": torch.__version__,
            "cuda": "12.6", "device_name": "NVIDIA A100-SXM4-40GB",
            "packages": {
                "torch": torch.__version__, "numpy": "1.26.4", "pandas": "2.2.2",
                "pyarrow": "18.1.0", "transformers": "4.48.0",
            },
        },
        "steps": [],
    })
    write_json(run / "scale_analysis_v1/gate/scale_gate.json", {"proceed": gate})
    return drive


def input_bundle(tmp_path: Path) -> Path:
    drive = drive_fixture(tmp_path)
    output = tmp_path / "exports"
    manifest = curc.export_input(drive, ROOT, output)
    return output / manifest["bundle_id"]


def value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_trailblazer_amendment_is_linked_and_additive() -> None:
    protocol = StudyProtocol.from_path(ROOT / f"studies/{curc.STUDY}/protocol.json")
    amendment = ROOT / f"studies/{curc.STUDY}/amendments/{curc.AMENDMENT_NAME}"
    no_memory = protocol.run_spec("medium_no_memory_e25_v1", amendment_paths=[amendment])
    assert no_memory["memory_mode"] == "no_memory"
    assert no_memory["require_panel_completion"] is True
    assert protocol.run_spec("medium_adaptive_e50_milestone_v1", amendment_paths=[amendment])["budget_bases"] == 50_000_000
    replica = protocol.run_spec("medium_adaptive_seed2_e100_increment_v1", amendment_paths=[amendment])
    assert replica["seed"] == curc.REPLICA_SEED
    assert replica["trajectory_id"] == "adaptive-seed-20260752"
    assert replica["warm_start_from"] == "medium_adaptive_seed2_e25_v1"
    failover = protocol.run_spec("medium_adaptive_gcp_failover_chunk_v1", amendment_paths=[amendment])
    assert failover["seed"] == curc.PRIMARY_SEED and failover["execution_role"] == "gcp_failover"
    assert "medium_adaptive_e50_milestone_v1" not in protocol.payload["run_matrix"]


def test_export_requires_completed_passing_e25_gate(tmp_path: Path) -> None:
    drive = drive_fixture(tmp_path, gate=False)
    with pytest.raises(curc.ContractError, match="does not authorize"):
        curc.export_input(drive, ROOT, tmp_path / "out")
    assert not list((tmp_path / "out").glob("*")) if (tmp_path / "out").exists() else True


def test_curc_commands_preserve_trailblazer_configuration(tmp_path: Path) -> None:
    bundle = input_bundle(tmp_path)
    python = tmp_path / "venv/bin/python"
    first = curc.training_command(
        bundle, tmp_path / "run", python, cap_bases=curc.E50_BASES, first_chunk=True,
    )
    expected = {
        "--memory-mode": "adaptive", "--horizon": "3", "--seed": "20260751",
        "--scheduler-policy": "stateful_rotation", "--scheduler-burst-segments": "96",
        "--checkpoint-every": "250", "--learning-rate": "3e-5",
        "--min-learning-rate": "3e-6", "--lr-warmup-bases": "2000000",
        "--lr-decay-bases": "100000000", "--weight-decay": "0.1",
        "--gradient-clip-norm": "0.5", "--block-count": "12", "--d-model": "256",
        "--num-heads": "8", "--memory-depth": "2", "--max-valid-bases": "50000000",
        "--run-id": "medium_adaptive_e50_milestone_v1",
    }
    for flag, value in expected.items():
        assert value_after(first, flag) == value
    assert "--warm-start-checkpoint" in first and "--no-resume" in first
    assert "--require-panel-completion" not in first
    final = curc.training_command(bundle, tmp_path / "run", python, final=True)
    assert "--require-panel-completion" in final
    assert "--warm-start-checkpoint" not in final and "--no-resume" not in final
    assert value_after(final, "--run-id") == curc.E100_RUN_ID


def test_execution_plan_respects_walltime_and_scientific_milestones() -> None:
    chunks = curc.execution_chunks(26_062_903, 101_062_903, 53.919728, 160)
    milestones = [item["milestone"] for item in chunks if item["milestone"]]
    assert milestones == ["e50", "e75", "e100"]
    assert chunks[-1]["final"] is True and chunks[-1]["cap_bases"] is None
    assert all(item["index"] == index for index, item in enumerate(chunks))
    safe = int(53.919728 * 160 * 3600 * 0.85)
    positions = [26_062_903]
    for item in chunks:
        positions.append(101_062_903 if item["final"] else item["cap_bases"])
    assert max(b - a for a, b in zip(positions, positions[1:])) <= safe


def test_milestone_packaging_is_immutable_and_requires_complete_e100(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_checkpoint(run / "latest.pt", step=70_000, bases=75_000_100)
    write_json(run / "LIVE_STATUS.json", {"state": "completed"})
    manifest = curc.package_milestone(run, tmp_path / "out", "e75")
    bundle = tmp_path / "out" / manifest["bundle_id"]
    assert curc.verify_milestone(bundle)["checkpoint"]["processed_bases"] == 75_000_100
    assert (bundle / "model.pt").is_file()
    assert curc.sha256_file(bundle / "model.pt") == curc.sha256_file(bundle / "latest.pt")
    with pytest.raises(curc.ContractError, match="already exists"):
        curc.package_milestone(run, tmp_path / "out", "e75")
    with pytest.raises(curc.ContractError, match="run manifest is missing"):
        curc.package_milestone(run, tmp_path / "other", "e100")


def test_curc_recovery_is_immutable_and_preserves_primary_identity(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_checkpoint(run / "latest.pt", step=55_000, bases=50_500_000)
    write_json(run / "LIVE_STATUS.json", {"state": "running"})
    manifest = curc.package_recovery(run, tmp_path / "recovery")
    bundle = tmp_path / "recovery" / manifest["bundle_id"]
    verified = curc.verify_recovery(bundle)
    assert verified["trajectory_id"] == "adaptive-seed-20260751"
    assert verified["execution_role"] == "curc-primary"
    assert verified["checkpoint"]["optimizer_step"] == 55_000
    with pytest.raises(curc.ContractError, match="already exists"):
        curc.package_recovery(run, tmp_path / "recovery")
    write_checkpoint(run / "latest.pt", step=55_001, bases=50_500_576, seed=curc.REPLICA_SEED)
    with pytest.raises(curc.ContractError, match="scheduler seed"):
        curc.package_recovery(run, tmp_path / "other")


def test_adaptive_gcp_roles_separate_failover_and_replica(tmp_path: Path) -> None:
    bundle = input_bundle(tmp_path)
    replica = adaptive_gcp.role_contract(bundle, adaptive_gcp.ROLE_REPLICA)
    assert replica["scientific_seed"] == curc.REPLICA_SEED
    assert replica["recovery"] is None
    with pytest.raises(adaptive_gcp.ContractError, match="requires a CURC recovery"):
        adaptive_gcp.role_contract(bundle, adaptive_gcp.ROLE_FAILOVER)
    run = tmp_path / "curc-run"
    write_checkpoint(run / "latest.pt", step=55_000, bases=50_500_000)
    recovery_manifest = curc.package_recovery(run, tmp_path / "recoveries")
    recovery = tmp_path / "recoveries" / recovery_manifest["bundle_id"]
    failover = adaptive_gcp.role_contract(bundle, adaptive_gcp.ROLE_FAILOVER, recovery)
    assert failover["scientific_seed"] == curc.PRIMARY_SEED
    assert failover["trajectory_id"] != replica["trajectory_id"]
    with pytest.raises(adaptive_gcp.ContractError, match="must not inherit"):
        adaptive_gcp.role_contract(bundle, adaptive_gcp.ROLE_REPLICA, recovery)


def test_adaptive_gcp_replica_command_is_independently_seeded(tmp_path: Path) -> None:
    bundle = input_bundle(tmp_path)
    run_root = tmp_path / "replica"
    command = adaptive_gcp.training_command(
        bundle, run_root, tmp_path / "venv/bin/python",
        role=adaptive_gcp.ROLE_REPLICA, stage="e25",
    )
    assert value_after(command, "--seed") == str(curc.REPLICA_SEED)
    assert value_after(command, "--memory-mode") == "adaptive"
    assert value_after(command, "--panel-manifest").endswith("panels/e25.json")
    assert value_after(command, "--run-id") == "medium_adaptive_seed2_e25_v1"
    assert "--require-panel-completion" in command
    assert "--warm-start-checkpoint" not in command
    write_checkpoint(
        run_root / "e25/latest.pt", step=40_000, bases=26_062_903,
        seed=curc.REPLICA_SEED,
    )
    e100 = adaptive_gcp.training_command(
        bundle, run_root, tmp_path / "venv/bin/python",
        role=adaptive_gcp.ROLE_REPLICA, stage="e50",
    )
    assert value_after(e100, "--seed") == str(curc.REPLICA_SEED)
    assert value_after(e100, "--max-valid-bases") == str(curc.E50_BASES)
    assert "--warm-start-checkpoint" in e100 and "--no-resume" in e100


def test_adaptive_gcp_staging_never_overwrites_newer_state(tmp_path: Path) -> None:
    source_run = tmp_path / "curc"
    write_checkpoint(source_run / "latest.pt", step=55_000, bases=50_500_000)
    recovery_manifest = curc.package_recovery(source_run, tmp_path / "recoveries")
    recovery = tmp_path / "recoveries" / recovery_manifest["bundle_id"]
    root = tmp_path / "gcp"
    assert adaptive_gcp.stage_recovery(recovery, root)["state"] == "staged"
    write_checkpoint(root / "e100/latest.pt", step=56_000, bases=51_000_000)
    assert adaptive_gcp.stage_recovery(recovery, root)["state"] == "kept_newer_or_equal"


def test_credit_activation_is_private_and_grant_gated(tmp_path: Path) -> None:
    activation = tmp_path / "GOOGLE_CLOUD_CREDIT_ACTIVATION.json"
    write_json(activation, {
        "active": True, "project": "divine-tempo-502518-j4", "award_usd": 5000,
        "award_id": "test-award", "billing_account_id": "000000-000000-000000",
        "gpu_hour_allocations": {"gcp-failover": 450, "gcp-replica": 582},
    })
    assert adaptive_gcp.verify_credit_activation(
        activation, "divine-tempo-502518-j4", "gcp-replica",
    )["award_id"] == "test-award"
    with pytest.raises(adaptive_gcp.ContractError, match="not confirmed active"):
        adaptive_gcp.verify_credit_activation(activation, "wrong-project")
    write_json(activation, {
        "active": True, "project": "divine-tempo-502518-j4", "award_usd": 5000,
        "award_id": "test-award", "billing_account_id": "000000-000000-000000",
        "gpu_hour_allocations": {"gcp-failover": 700, "gcp-replica": 700},
    })
    with pytest.raises(adaptive_gcp.ContractError, match="campaign cap"):
        adaptive_gcp.verify_credit_activation(activation, "divine-tempo-502518-j4")


def test_failover_authorization_enforces_the_fourteen_day_trigger(tmp_path: Path) -> None:
    authorization = tmp_path / "FAILOVER_AUTHORIZATION.json"
    write_json(authorization, {
        "authorized": True, "reason": "curc_not_started_14_days",
        "authorized_by": "researcher", "curc_planned_start_at": "2026-10-15T00:00:00-06:00",
        "authorized_at": "2026-10-29T00:00:00-06:00",
    })
    assert adaptive_gcp.verify_failover_authorization(authorization)["authorized"] is True
    write_json(authorization, {
        "authorized": True, "reason": "curc_not_started_14_days",
        "authorized_by": "researcher", "curc_planned_start_at": "2026-10-15T00:00:00-06:00",
        "authorized_at": "2026-10-28T23:59:59-06:00",
    })
    with pytest.raises(adaptive_gcp.ContractError, match="has not elapsed"):
        adaptive_gcp.verify_failover_authorization(authorization)
    write_json(authorization, {
        "authorized": True, "reason": "curc_nonrecoverable_failure",
        "authorized_by": "researcher", "authorized_at": "2026-10-20T00:00:00-06:00",
    })
    with pytest.raises(adaptive_gcp.ContractError, match="evidence"):
        adaptive_gcp.verify_failover_authorization(authorization)


def test_gcp_no_memory_command_has_one_scientific_mode_difference(tmp_path: Path) -> None:
    bundle = input_bundle(tmp_path)
    command = control.training_command(bundle, tmp_path / "control", tmp_path / "venv/bin/python")
    assert value_after(command, "--memory-mode") == "no_memory"
    assert value_after(command, "--run-id") == control.RUN_ID
    assert "--require-panel-completion" in command
    assert "--warm-start-checkpoint" not in command and "--no-resume" not in command
    for flag, expected in {
        "--seed": "20260751", "--horizon": "3", "--scheduler-burst-segments": "96",
        "--learning-rate": "3e-5", "--min-learning-rate": "3e-6",
        "--lr-decay-bases": "100000000", "--block-count": "12", "--d-model": "256",
        "--num-heads": "8", "--memory-depth": "2",
    }.items():
        assert value_after(command, flag) == expected


def test_gcp_cost_projection_preserves_reserve_and_rejects_bad_inputs() -> None:
    affordable = control.cost_projection(26_062_903, 150.0)
    assert affordable["within_working_budget"] is True
    expensive = control.cost_projection(26_062_903, 50.0)
    assert expensive["within_working_budget"] is False
    with pytest.raises(control.ContractError):
        control.cost_projection(26_062_903, 0)
    cap = control.cumulative_runtime_cap(70.0, control.DEFAULT_HOURLY_USD, 20.0)
    assert cap * control.DEFAULT_HOURLY_USD + 20.0 <= control.WORKING_BUDGET_USD


def test_gcp_runtime_ledger_is_cumulative_and_counts_interrupted_sessions(tmp_path: Path) -> None:
    ledger, first = control._begin_runtime_session(tmp_path, 50.0)
    assert first == pytest.approx(35.75)
    control._finish_runtime_session(ledger, 1.5, "pilot_completed")
    _, second = control._begin_runtime_session(tmp_path, 50.0)
    assert second == pytest.approx(35.75)
    # Simulate a hard VM stop: a new session conservatively charges the full
    # prior declared allowance before calculating what remains.
    _, third = control._begin_runtime_session(tmp_path, 50.0)
    assert third == pytest.approx(12.75)
    payload = json.loads(ledger.read_text())
    assert payload["consumed_hours"] == pytest.approx(37.25)
    assert payload["sessions"][-2]["state"] == "interrupted"


def test_generated_notebooks_are_parameterized_and_leave_03l_unchanged() -> None:
    handoff = json.loads((ROOT / "notebooks/titans_stage_c/03r_stage_c_v3_medium_adaptive_e100_curc_handoff.ipynb").read_text())
    evaluation = json.loads((ROOT / "notebooks/titans_stage_c/03s_stage_c_v3_medium_adaptive_milestone_evaluation.ipynb").read_text())
    handoff_text = "".join("".join(cell["source"]) for cell in handoff["cells"])
    evaluation_text = "".join("".join(cell["source"]) for cell in evaluation["cells"])
    assert "divine-tempo-502518-j4" in handoff_text and "BUCKET='ecoeus'" in handoff_text
    assert "TRAILBLAZER_INPUT_MANIFEST.json" in handoff_text
    assert "gcp-replica" in handoff_text and "gcp-failover" in handoff_text
    assert "export-recovery" in handoff_text and "model.pt" in handoff_text
    assert "MILESTONE='e50'" in evaluation_text
    assert "MODEL_PATH=''" in evaluation_text
    assert "context_smoke" in evaluation_text and "context_full" in evaluation_text
    for notebook in (handoff, evaluation):
        for index, notebook_cell in enumerate(notebook["cells"]):
            if notebook_cell["cell_type"] == "code":
                ast.parse("".join(notebook_cell["source"]), filename=f"cell-{index}")
    assert (ROOT / "notebooks/titans_stage_c/03l_stage_c_v3_medium_adaptive_e25.ipynb").is_file()


def test_google_research_credits_application_is_copy_ready_and_secret_free() -> None:
    application = (
        ROOT / "docs/titans_stage_c/GOOGLE_CLOUD_RESEARCH_CREDITS_APPLICATION.md"
    ).read_text()
    proposal = application.split("## Proposal (235 words)", 1)[1].split(
        "## Pricing Calculator recipe", 1,
    )[0]
    words = proposal.replace("\n", " ").split()
    assert len(words) == 235
    assert "divine-tempo-502518-j4" in application and "`ecoeus`" in application
    assert "PRIVATE — enter only in the online form" in application
    assert "000000-000000-000000" not in application
    assert "kickstart_your_research_with_google_cloud_credits_tips_on_applying.pdf" in application
