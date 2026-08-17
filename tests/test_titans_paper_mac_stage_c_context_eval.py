from __future__ import annotations

import json
from pathlib import Path
import random
import shutil

import pytest

from seqtrainer.torch.titans_paper_mac_stage_c.context_eval import (
    CaseResumeStore,
    ContextEvalConfig,
    TokenStreamSlice,
    association_occurrences,
    average_precision,
    boundary_metrics,
    context_conflict,
    detection_metrics,
    materialize_anomaly_sequences,
    materialize_needle_sequence,
    needle_metrics,
    paired_host_bootstrap,
    select_anomaly_cases,
    select_needle_cases,
    sha256_file,
    stage_checkpoint,
    threshold_at_fpr,
    write_jsonl,
    write_sequence_slices,
)
from seqtrainer.torch.titans_paper_mac_stage_c.tokenizers import SeqTrainerBaseTokenizer


def _stream(identity: str, accession: str, group: str, seed: int) -> TokenStreamSlice:
    rng = random.Random(seed)
    # Natural-looking, deterministic base tokens. Force the prospective write
    # association to a unique six-token pattern.
    tokens = [rng.randrange(2, 6) for _ in range(42 * 32 + 1)]
    write = 16 * 32
    tokens[write : write + 6] = [2, 2, 3, 3, 4, 5]
    pattern = tokens[write : write + 6]
    for index in range(len(tokens) - 5):
        if index != write and tokens[index : index + 6] == pattern:
            tokens[index] = 5 if tokens[index] != 5 else 4
    tokenizer = SeqTrainerBaseTokenizer()
    dna = tokenizer.decode(tokens)
    return TokenStreamSlice(identity, accession, group, tuple(tokens), (1,) * len(tokens), dna)


def _streams() -> tuple[TokenStreamSlice, ...]:
    return (
        _stream("host-a:chromosome", "host-a", "ani99:a", 10),
        _stream("host-b:chromosome", "host-b", "ani99:b", 20),
        _stream("host-c:chromosome", "host-c", "ani99:c", 30),
        _stream("host-d:chromosome", "host-d", "ani99:d", 40),
    )


def test_anomaly_selection_is_deterministic_cross_ani_aligned_and_gc_matched() -> None:
    config = ContextEvalConfig(gc_tolerance=0.20)
    first = select_anomaly_cases(_streams(), config)
    second = select_anomaly_cases(tuple(reversed(_streams())), config)
    assert first == second
    assert len(first) == 4
    lookup = {item.stream_id: item for item in _streams()}
    for case in first:
        assert case.host_accession != case.donor_accession
        assert case.host_clade_group != case.donor_clade_group
        assert case.insertion_segment == 16
        assert case.length_segments in {1, 4}
        assert abs(case.donor_gc - case.hard_negative_gc) <= config.gc_tolerance
        variants = materialize_anomaly_sequences(case, lookup)
        assert set(variants) == {"different_ani", "same_host", "untouched"}
        assert len({len(value) for value in variants.values()}) == 1
        start = case.insertion_segment * 32
        assert variants["different_ani"][:start] == variants["untouched"][:start]


def test_anomaly_selection_refuses_to_relax_gc_or_group_contract() -> None:
    streams = tuple(
        TokenStreamSlice(item.stream_id, item.accession, "same", item.token_ids, item.base_lengths, item.dna)
        for item in _streams()
    )
    with pytest.raises(ValueError, match="could not construct"):
        select_anomaly_cases(streams, ContextEvalConfig())


def test_anomaly_selection_searches_for_a_complete_multilength_pair() -> None:
    tokenizer = SeqTrainerBaseTokenizer()

    def segmented(identity: str, accession: str, group: str, values: list[int]) -> TokenStreamSlice:
        tokens = tuple(token for value in values for token in [value] * 32) + (values[-1],)
        return TokenStreamSlice(
            identity, accession, group, tokens, (1,) * len(tokens), tokenizer.decode(tokens)
        )

    host_pattern = ([2, 2, 4, 4] * 10)[:40]
    donor_pattern = [4, 2, 2, 4, 2, 2, 4, 4]
    streams = (
        segmented("host:chromosome", "host", "ani99:host", host_pattern),
        segmented("donor:chromosome", "donor", "ani99:donor", donor_pattern),
        segmented("wrong:chromosome", "wrong", "ani99:wrong", [4] * 16),
    )
    config = ContextEvalConfig(
        hosts=1,
        insertion_segments=(1, 4),
        needle_distances=(3,),
        gc_tolerance=0.0,
    )

    cases = select_anomaly_cases(streams, config)

    assert len(cases) == 2
    assert {case.donor_start_segment for case in cases} == {4}
    assert all(case.host_accession == "host" for case in cases)


def test_needle_cases_are_unique_natural_and_cover_smoke_grid() -> None:
    config = ContextEvalConfig(gc_tolerance=0.20)
    streams = _streams()
    cases = select_needle_cases(streams, config)
    lookup = {item.stream_id: item for item in streams}
    assert len(cases) == 4
    assert {(case.distance_segments, case.distractor_count) for case in cases} == {(3, 0), (16, 0)}
    for case in cases:
        host = lookup[case.host_stream_id]
        write = case.write_segment * 32
        assert association_occurrences(host.token_ids, case.key_tokens, case.value_tokens) == [write]
        sequence = materialize_needle_sequence(case, lookup)
        query = case.query_segment * 32
        assert sequence[query : query + 4] == case.key_tokens
        assert sequence[query + 4 : query + 6] == case.value_tokens

    interference = ContextEvalConfig(
        hosts=1, insertion_segments=(1,), needle_distances=(3,),
        distractor_counts=(16,), gc_tolerance=0.20,
    )
    crowded = select_needle_cases(streams, interference)[0]
    crowded_sequence = materialize_needle_sequence(crowded, lookup)
    assert association_occurrences(
        crowded_sequence, crowded.key_tokens, crowded.value_tokens
    ) == [crowded.write_segment * 32, crowded.query_segment * 32]


def test_detection_boundary_bootstrap_and_needle_statistics() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert average_precision(labels, scores) == 1.0
    assert threshold_at_fpr([0.1, 0.2], 0.0) > 0.2
    metrics = detection_metrics(labels, scores, represented_bases=[100] * 4)
    assert metrics["auprc"] == 1.0
    assert metrics["tpr_at_0.01"] == 1.0
    assert boundary_metrics([0.0, 0.8, 0.9, 0.1], boundary=1, threshold=0.5, positive_end=3) == {
        "detection_delay": 0, "boundary_error": 0, "recovery_time": 0
    }
    assert context_conflict(1.25, 1.0) == pytest.approx(0.25)
    bootstrap = paired_host_bootstrap({"a": [1.0], "b": [3.0]}, seed=2, samples=100)
    assert bootstrap["mean"] == 2.0
    rows = [
        {"host_accession": host, "distance_segments": distance, "distractor_count": distractors,
         "carried_log_probability": 2.0, "reset_log_probability": 1.0}
        for host in "abcdefgh" for distance in (16, 32) for distractors in (0, 4)
    ]
    summary = needle_metrics(rows, seed=3, samples=100)
    assert summary["promising_long_context_signal"] is True
    assert summary["distance_decay"] == {"16": 1.0, "32": 1.0}


def test_checkpoint_staging_is_immutable_and_detects_source_change(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"owned checkpoint")
    registry = tmp_path / "registry"
    metadata = {"optimizer_step": 19, "processed_bases": 123, "code_commit": "abc"}
    staged = stage_checkpoint(source, registry, metadata, trusted=True)
    assert staged.name == f"19_{sha256_file(source)[:12]}"
    assert (staged / "checkpoint.pt").read_bytes() == source.read_bytes()
    assert stage_checkpoint(source, registry, metadata, trusted=True) == staged
    with pytest.raises(ValueError, match="trust"):
        stage_checkpoint(source, tmp_path / "other", metadata, trusted=False)

    changing = tmp_path / "changing.pt"
    changing.write_bytes(b"before")

    def mutate_after_copy(src: Path, dst: Path) -> object:
        result = shutil.copyfile(src, dst)
        src.write_bytes(b"after")
        return result

    with pytest.raises(RuntimeError, match="changed during copying"):
        stage_checkpoint(changing, tmp_path / "changed_registry", metadata, trusted=True, copy=mutate_after_copy)


def test_atomic_resume_and_retained_artifacts_are_reproducible(tmp_path: Path) -> None:
    contract = {"checkpoint": "sha", "cases": "case-sha"}
    store = CaseResumeStore(tmp_path / "resume", contract)
    path = store.commit("case-a", {"finite": 1.0})
    assert path.is_file() and store.completed("case-a")
    assert store.commit("case-a", {"finite": 1.0}) == path
    with pytest.raises(ValueError, match="completed case changed"):
        store.commit("case-a", {"finite": 2.0})
    with pytest.raises(ValueError, match="contract changed"):
        CaseResumeStore(tmp_path / "resume", {"checkpoint": "different"})

    jsonl = write_jsonl(tmp_path / "cases.jsonl", [{"b": 2, "a": 1}])
    assert jsonl.read_text() == '{"a":1,"b":2}\n'
    fasta = write_sequence_slices(tmp_path / "slices.fasta.gz", [("used", "ACGT")])
    assert fasta.is_file()
    assert not list(tmp_path.rglob("*.partial"))


def test_notebook_is_five_cell_bounded_c19_first_validation_study() -> None:
    path = Path("notebooks/titans_stage_c/03q_stage_c_c19_context_anomaly_and_needle.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 5
    assert "C16_CHECKPOINT" in source and "C19_CHECKPOINT" in source
    assert "21898362291f4fd1e6aafcfbe47e8b05dbe69e5c8036e6ae7927a6ac24ac4541" in source
    assert "07fb2069b1f29a76898a90d8dfb899c5ca46cb90608fac45bc0ddff9876dbd1a" in source
    assert "E25_TRAINING_PANEL" in source and "ANI_PAIRS" in source
    assert "titans_paper_mac_stage_c.anomaly_study_cli" in source
    for dependency in ("scikit-learn>=1.3,<2", "pyarrow==18.1.0", "scipy>=1.11,<2", "pytest>=8,<9"):
        assert dependency in source
    assert "RUN_C19=True" in source
    assert "RUN_C16_COMPARISON=False" in source
    assert "MAX_C19_HOURS=22.0" in source
    assert "--mode','bounded'" in source
    assert "total_segment_forwards']!=25344" in source
    assert "estimate-runtime" in source
    assert "--max-runtime-hours" in source
    assert source.index("run_model('C19'") < source.index("run_model('C16'")
    assert "def initialize_drive():" in source
    assert "force_remount=bool(attempt)" in source
    assert "drive.flush_and_unmount()" in source
    assert "'.03q_drive_write_probe'" in source
    assert "errno.EIO,errno.ESTALE,errno.ENOTCONN" in source
    assert "RUN_LOCKED_TEST" not in source
    assert "--run-locked-test" not in source
    assert "--e25-panel" in source and "--ani-membership" in source
    assert "2e869da44c2fb00c93101f72dfe7a88074fb4e2c" in source
    assert "REPLACE_WITH_EVALUATOR_COMMIT" not in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"03q-cell-{index}", "exec")
