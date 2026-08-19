from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from seqtrainer.data.bacteria_titan import (
    StageCPanelManifest,
    TokenStreamDataset,
    build_panel_stream_cache,
    materialize_token_stream_dataset,
    resolved_parent_dataset_fingerprint,
    validate_panel_against_dataset,
    validate_panel_stream_cache,
)
from seqtrainer.torch.titans_paper_mac_stage_c.tokenizers import SeqTrainerBaseTokenizer


def _source(tmp_path: Path) -> tuple[Path, list[str]]:
    records = []
    for index, length in enumerate((20, 22, 24, 26, 28, 30)):
        records.append({
            "accession": f"GCF_{index + 1:09d}.1",
            "contig_id": "chromosome",
            "sequence": ("ACGT" * 20)[:length],
            "split": "val" if index % 2 == 0 else "train",
            "clade_group": f"ani99:{index}",
        })
    root = tmp_path / "source"
    materialize_token_stream_dataset(
        records, SeqTrainerBaseTokenizer(), root, tokens_per_shard=50
    )
    return root, [f"{row['accession']}:chromosome" for row in records]


def _panel(path: Path, source: Path, stream_ids: list[str], split: str, role: str) -> Path:
    dataset = TokenStreamDataset(source)
    rows = {row.stream_id: row for row in dataset.index}
    predictable = sum(
        rows[value].base_count
        - int(dataset.base_lengths[rows[value].shard_index][rows[value].token_offset])
        for value in stream_ids
    )
    payload = {
        "format_version": 1,
        "panel_id": path.stem,
        "role": role,
        "split": split,
        "parent_dataset_fingerprint": resolved_parent_dataset_fingerprint(source),
        "stream_ids": stream_ids,
        "accessions": [rows[value].accession for value in stream_ids],
        "predictable_bases": predictable,
        "selection_order": "test",
    }
    path.write_text(json.dumps(payload) + "\n")
    return path


def test_cache_is_exact_deterministic_and_parent_compatible(tmp_path: Path) -> None:
    source, ids = _source(tmp_path)
    selected = [ids[4], ids[1], ids[2]]
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = build_panel_stream_cache(source, first, stream_ids=selected)
    build_panel_stream_cache(source, second, stream_ids=reversed(selected))

    assert manifest["streams"] == 3
    assert manifest["source_shard_indices"] == [0, 1, 2]
    assert (first / "token_stream_manifest.json").read_bytes() == (
        second / "token_stream_manifest.json"
    ).read_bytes()
    assert (first / "token_stream_index.jsonl").read_bytes() == (
        second / "token_stream_index.jsonl"
    ).read_bytes()
    original = TokenStreamDataset(source, verify_checksums=True)
    compact = TokenStreamDataset(first, verify_checksums=True)
    original_rows = {row.stream_id: row for row in original.index}
    compact_rows = {row.stream_id: row for row in compact.index}
    assert set(compact_rows) == set(selected)
    for stream_id in selected:
        left, right = original_rows[stream_id], compact_rows[stream_id]
        assert np.array_equal(
            original.tokens[left.shard_index][left.token_offset:left.token_offset + left.token_count],
            compact.tokens[right.shard_index][right.token_offset:right.token_offset + right.token_count],
        )
        assert np.array_equal(
            original.base_lengths[left.shard_index][left.token_offset:left.token_offset + left.token_count],
            compact.base_lengths[right.shard_index][right.token_offset:right.token_offset + right.token_count],
        )
    assert resolved_parent_dataset_fingerprint(first) == resolved_parent_dataset_fingerprint(source)


def test_cache_skips_valid_chunks_and_rebuilds_corruption(tmp_path: Path, monkeypatch) -> None:
    source, ids = _source(tmp_path)
    output = tmp_path / "cache"
    build_panel_stream_cache(source, output, stream_ids=(ids[0], ids[2], ids[4]))
    mtimes = {
        path: path.stat().st_mtime_ns for path in output.glob("chunks/*/tokens.npy")
    }
    build_panel_stream_cache(source, output, stream_ids=(ids[4], ids[2], ids[0]))
    assert mtimes == {path: path.stat().st_mtime_ns for path in mtimes}

    damaged = sorted(mtimes)[1]
    damaged.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="chunk"):
        validate_panel_stream_cache(output)
    build_panel_stream_cache(source, output, stream_ids=(ids[0], ids[2], ids[4]))
    validate_panel_stream_cache(output)
    assert damaged.stat().st_size > len(b"corrupt")


def test_compact_panel_validation_detects_split_offset_hash_and_base_changes(tmp_path: Path) -> None:
    source, ids = _source(tmp_path)
    val_ids = [ids[0], ids[2], ids[4]]
    panel_path = _panel(tmp_path / "validation.json", source, val_ids, "val", "validation")
    cache = tmp_path / "cache"
    build_panel_stream_cache(source, cache, stream_ids=val_ids)
    panel = StageCPanelManifest.from_path(panel_path)
    validate_panel_against_dataset(panel, TokenStreamDataset(cache, verify_checksums=True))

    index_path = cache / "token_stream_index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    rows[0]["token_offset"] += 1
    index_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="index checksum"):
        TokenStreamDataset(cache, verify_checksums=True)


def test_interruption_preserves_completed_chunks_and_resumes(tmp_path: Path, monkeypatch) -> None:
    source, ids = _source(tmp_path)
    output = tmp_path / "cache"
    real_copy = shutil.copyfile
    calls = 0

    def interrupt(source_path, destination_path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 6:
            raise OSError("simulated Drive disconnection")
        return real_copy(source_path, destination_path, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", interrupt)
    with pytest.raises(OSError, match="disconnection"):
        build_panel_stream_cache(source, output, stream_ids=(ids[0], ids[2], ids[4]))
    chunks = sorted((output / "chunks").iterdir())
    assert (chunks[0] / "COMPLETE.json").is_file()
    assert not (chunks[1] / "COMPLETE.json").exists()
    assert not list(output.rglob("*.partial"))

    monkeypatch.setattr(shutil, "copyfile", real_copy)
    build_panel_stream_cache(source, output, stream_ids=(ids[4], ids[0], ids[2]))
    validate_panel_stream_cache(output)
