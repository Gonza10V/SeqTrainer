from __future__ import annotations

import pandas as pd
import pytest

from seqtrainer.torch.titans_paper_mac_stage_c.anomaly_study import (
    ANOMALY_LENGTHS, BOUNDED_ANOMALY_LENGTHS, BOUNDED_NEEDLE_DISTANCES,
    INSERTION_DEPTHS, ScientificStudyConfig, analysis_contract,
    canonical_host_calibration_start, choose_relative_donors, evaluation_interventions,
    exact_sign_permutation_p, holm_adjust, peak_enrichment, nested_leave_one_host_out,
    planned_segment_forwards, robust_z, runtime_projection,
)
from seqtrainer.torch.titans_paper_mac_stage_c.anomaly_study_cli import (
    available_analysis_models, parse_args, runtime_deadline_reached,
)
from seqtrainer.torch.titans_paper_mac_stage_c.context_eval import TokenStreamSlice


def test_full_contract_is_frozen_and_excludes_test_panel() -> None:
    config = ScientificStudyConfig.full()
    assert config.hosts == 8
    assert config.lengths == ANOMALY_LENGTHS == (1, 4, 16, 64)
    assert config.depths == INSERTION_DEPTHS == (16, 128)
    assert config.native_calibration_segments == 128
    assert analysis_contract()["test_panel"] == "excluded"
    with pytest.raises(ValueError, match="exact preregistered"):
        ScientificStudyConfig(mode="full", hosts=8)


def test_bounded_contract_has_exact_sub_24_hour_workload() -> None:
    config = ScientificStudyConfig.bounded()
    assert config.hosts == 8
    assert config.lengths == BOUNDED_ANOMALY_LENGTHS == (1, 16, 64)
    assert config.depths == (16,)
    assert config.needle_distances == BOUNDED_NEEDLE_DISTANCES == (3, 16, 64)
    assert config.needle_distractors == (0, 16)
    workload = planned_segment_forwards(config)
    assert workload == {
        "anomaly_cases": 48,
        "anomaly_scoring": 14688,
        "anomaly_warmup": 768,
        "native_calibration": 1024,
        "needle_scoring_and_warmup": 8864,
        "total_segment_forwards": 25344,
    }
    assert analysis_contract()["bounded_workload"] == workload
    assert evaluation_interventions("bounded", "foreign_near") == (
        "carried", "reset", "wrong_host", "no_memory",
    )
    assert evaluation_interventions("bounded", "same_host") == ("carried",)
    assert len(evaluation_interventions("full", "untouched")) == 4
    with pytest.raises(ValueError, match="sub-24-hour"):
        ScientificStudyConfig(
            mode="bounded", hosts=8, lengths=(1, 4, 16, 64), depths=(16,),
            needle_distances=(3, 16, 64), needle_distractors=(0, 16),
        )


def test_runtime_projection_uses_slower_rate_and_fails_closed() -> None:
    accepted = runtime_projection(
        planned_forwards=25344, measured_segments_per_second=1.2,
    )
    assert accepted["effective_segments_per_second"] == pytest.approx(0.474)
    assert accepted["projected_hours"] == pytest.approx(18.3677918)
    assert accepted["accepted"] is True
    refused = runtime_projection(
        planned_forwards=25344, measured_segments_per_second=0.30,
    )
    assert refused["projected_hours"] > 22
    assert refused["accepted"] is False


def test_canonical_host_layout_rejects_prefix_n_and_selects_middle_calibration() -> None:
    config = ScientificStudyConfig.bounded()
    segment_tokens = 32
    complete_segments = 260
    token_count = complete_segments * segment_tokens + 1

    def stream(identity: str, noncanonical_positions: set[int]) -> TokenStreamSlice:
        dna = ["A"] * token_count
        for position in noncanonical_positions:
            dna[position] = "N"
        return TokenStreamSlice(
            identity,
            identity,
            f"ani99:{identity}",
            tuple([3] * token_count),
            tuple([1] * token_count),
            "".join(dna),
        )

    maximum_end = max(config.depths) + max(config.lengths) + config.recovery_segments
    prefix_bad = stream("prefix-bad", {10})
    assert canonical_host_calibration_start(prefix_bad, config) is None

    tail_bad_position = (complete_segments - 1) * segment_tokens
    tail_bad = stream("tail-bad", {tail_bad_position})
    assert canonical_host_calibration_start(tail_bad, config) == maximum_end

    no_window = stream(
        "no-window",
        {
            start * segment_tokens
            for start in range(maximum_end, complete_segments - config.native_calibration_segments + 1)
        },
    )
    assert canonical_host_calibration_start(no_window, config) is None


def test_analysis_is_c19_first_and_c16_is_optional(tmp_path) -> None:
    with pytest.raises(ValueError, match="primary C19"):
        available_analysis_models(tmp_path)
    (tmp_path / "C19").mkdir()
    (tmp_path / "C19" / "COMPLETE.json").write_text("{}\n")
    assert available_analysis_models(tmp_path) == ("C19",)
    (tmp_path / "C16").mkdir()
    (tmp_path / "C16" / "COMPLETE.json").write_text("{}\n")
    assert available_analysis_models(tmp_path) == ("C19", "C16")


def test_runtime_deadline_pauses_only_at_the_declared_boundary() -> None:
    assert runtime_deadline_reached(100.0, None, now_monotonic=10_000.0) is False
    assert runtime_deadline_reached(100.0, 2.0, now_monotonic=7_299.9) is False
    assert runtime_deadline_reached(100.0, 2.0, now_monotonic=7_300.0) is True


def test_run_model_dataset_arguments_are_optional_compatibility_checks(tmp_path) -> None:
    args = parse_args([
        "run-model", "--model", "C19", "--checkpoint", str(tmp_path / "model.pt"),
        "--frozen-panel", str(tmp_path / "panel"), "--output", str(tmp_path / "output"),
    ])
    assert args.dataset_dir is None
    assert args.validation_panel is None


def test_relative_donors_are_exact_e25_extremes_with_gap() -> None:
    ani = pd.DataFrame({
        "left": ["host", "host", "host"],
        "right": ["near", "middle", "far"],
        "ani": [98.8, 97.0, 95.2],
    })
    groups = {"host": "h", "near": "n", "middle": "m", "far": "f"}
    pair = choose_relative_donors(["host"], ["near", "middle", "far"], ani, groups)[0]
    assert (pair.near_accession, pair.near_ani) == ("near", 98.8)
    assert (pair.far_accession, pair.far_ani) == ("far", 95.2)


def test_host_statistics_and_circular_peak_are_deterministic() -> None:
    assert robust_z([1, 2, 3], [0, 1, 2]).tolist() == pytest.approx([0, 0.674490759, 1.348981518])
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert exact_sign_permutation_p([1, 1, 1]) == pytest.approx(0.25)
    result = peak_enrichment([0, 0, 10, 9, 0, 0], [False, False, True, True, False, False])
    assert result["maximum_inside_insert"] is True
    assert result["global_peak_index"] == 2


def test_nested_leave_one_host_out_keeps_calibration_inside_training_hosts() -> None:
    rows = []
    for host_index, host in enumerate("abc"):
        rows.extend({
            "host_accession": host, "is_foreign": False, "is_native_calibration": True,
            "bpb": 0.1 + index / 10_000, "markov_nll": 0.2, "tetranucleotide_distance": 0.1,
        } for index in range(60))
        rows.extend({
            "host_accession": host, "is_foreign": True, "is_native_calibration": False,
            "bpb": 2.0 + host_index / 100, "markov_nll": 1.5, "tetranucleotide_distance": 1.0,
        } for _ in range(10))
    predictions, coefficients = nested_leave_one_host_out(
        pd.DataFrame(rows), ("bpb", "markov_nll", "tetranucleotide_distance")
    )
    assert set(predictions["outer_host"]) == {"a", "b", "c"}
    assert predictions["training_native_threshold_1pct_fpr"].notna().all()
    assert set(coefficients["outer_host"]) == {"a", "b", "c"}
