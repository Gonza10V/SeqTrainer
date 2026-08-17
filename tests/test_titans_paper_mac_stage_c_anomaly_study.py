from __future__ import annotations

import pandas as pd
import pytest

from seqtrainer.torch.titans_paper_mac_stage_c.anomaly_study import (
    ANOMALY_LENGTHS, BOUNDED_ANOMALY_LENGTHS, BOUNDED_NEEDLE_DISTANCES,
    INSERTION_DEPTHS, ScientificStudyConfig, analysis_contract,
    choose_relative_donors, evaluation_interventions, exact_sign_permutation_p, holm_adjust, peak_enrichment,
    nested_leave_one_host_out, planned_segment_forwards, robust_z, runtime_projection,
)
from seqtrainer.torch.titans_paper_mac_stage_c.anomaly_study_cli import (
    available_analysis_models, runtime_deadline_reached,
)


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
