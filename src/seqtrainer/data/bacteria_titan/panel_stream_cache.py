"""Deterministic, resumable token-stream subsets for frozen Stage C panels."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .stage_c_streams import TokenStreamIndex


PANEL_STREAM_CACHE_FORMAT_VERSION = 1


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(_canonical(value) + b"\n")
    os.replace(partial, path)


def _read_index(path: Path) -> tuple[TokenStreamIndex, ...]:
    return tuple(
        TokenStreamIndex(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def panel_stream_ids(panel_paths: Sequence[str | Path]) -> frozenset[str]:
    """Return the exact union of stream IDs in one or more panel manifests."""

    selected: set[str] = set()
    for panel_path in panel_paths:
        payload = json.loads(Path(panel_path).read_text(encoding="utf-8"))
        values = payload.get("stream_ids")
        if not isinstance(values, list) or not values:
            raise ValueError(f"panel has no stream_ids: {panel_path}")
        selected.update(map(str, values))
    return frozenset(selected)


def _cache_contract(
    source_manifest_sha256: str,
    source_manifest: Mapping[str, object],
    selected: Iterable[str],
) -> dict[str, object]:
    return {
        "format_version": PANEL_STREAM_CACHE_FORMAT_VERSION,
        "layout": "source_shard_atomic_token_stream_subset",
        "parent_dataset_fingerprint": source_manifest_sha256,
        "tokenizer": source_manifest["tokenizer"],
        "segment_length": source_manifest["segment_length"],
        "stream_ids": sorted(set(map(str, selected))),
    }


def _chunk_contract(
    cache_contract_sha256: str,
    source_shard: Mapping[str, object],
    compact_shard_index: int,
    rows: Sequence[TokenStreamIndex],
) -> dict[str, object]:
    coordinates = [
        {
            "stream_id": row.stream_id,
            "source_shard_index": row.shard_index,
            "source_token_offset": row.token_offset,
            "token_count": row.token_count,
            "base_count": row.base_count,
        }
        for row in rows
    ]
    return {
        "format_version": PANEL_STREAM_CACHE_FORMAT_VERSION,
        "cache_contract_sha256": cache_contract_sha256,
        "source_shard_index": int(source_shard["shard_index"]),
        "compact_shard_index": compact_shard_index,
        "source_tokens_sha256": source_shard["tokens_sha256"],
        "source_base_lengths_sha256": source_shard["base_lengths_sha256"],
        "source_coordinates": coordinates,
        "source_coordinates_sha256": _hash(coordinates),
    }


def _validate_chunk(
    chunk: Path,
    expected_contract: Mapping[str, object],
    *,
    verify_arrays: bool = True,
) -> dict[str, object] | None:
    manifest_path = chunk / "chunk_manifest.json"
    sentinel_path = chunk / "COMPLETE.json"
    if not manifest_path.is_file() or not sentinel_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if manifest.get("contract") != dict(expected_contract):
        return None
    if sentinel.get("chunk_manifest_sha256") != _sha256_file(manifest_path):
        return None
    for name, digest in manifest.get("output_sha256", {}).items():
        path = chunk / name
        if not path.is_file() or (verify_arrays and _sha256_file(path) != digest):
            return None
    return manifest


def validate_panel_stream_cache(root: str | Path, *, verify_arrays: bool = True) -> dict[str, object]:
    """Validate a completed compact dataset and every independently saved chunk."""

    cache_root = Path(root)
    manifest_path = cache_root / "token_stream_manifest.json"
    contract_path = cache_root / "cache_contract.json"
    complete_path = cache_root / "COMPLETE.json"
    if not (manifest_path.is_file() and contract_path.is_file() and complete_path.is_file()):
        raise ValueError("panel stream cache is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    contract_hash = _hash(contract)
    if manifest.get("cache_contract_sha256") != contract_hash:
        raise ValueError("panel stream cache contract changed")
    if complete.get("manifest_sha256") != _sha256_file(manifest_path):
        raise ValueError("panel stream cache manifest changed")
    if manifest.get("parent_dataset_fingerprint") != contract.get("parent_dataset_fingerprint"):
        raise ValueError("panel stream cache parent fingerprint changed")
    index_path = cache_root / str(manifest["index"])
    if _sha256_file(index_path) != manifest["index_sha256"]:
        raise ValueError("panel stream cache index changed")
    rows = _read_index(index_path)
    if len(rows) != int(manifest["streams"]):
        raise ValueError("panel stream cache stream count changed")
    if len(manifest["shards"]) != len(manifest["source_shard_indices"]):
        raise ValueError("panel stream cache source-shard mapping changed")
    if [int(shard["shard_index"]) for shard in manifest["shards"]] != list(
        range(len(manifest["shards"]))
    ):
        raise ValueError("panel stream cache compact shard order changed")
    if {row.stream_id for row in rows} != set(contract["stream_ids"]):
        raise ValueError("panel stream cache selected streams changed")
    if sum(row.token_count for row in rows) != int(manifest["tokens"]):
        raise ValueError("panel stream cache token count changed")
    if sum(row.base_count for row in rows) != int(manifest["bases"]):
        raise ValueError("panel stream cache represented-base count changed")
    for shard, source_shard_index in zip(manifest["shards"], manifest["source_shard_indices"]):
        selected_rows = [row for row in rows if row.shard_index == int(shard["shard_index"])]
        expected = _chunk_contract(
            contract_hash,
            {
                "shard_index": source_shard_index,
                "tokens_sha256": shard["source_tokens_sha256"],
                "base_lengths_sha256": shard["source_base_lengths_sha256"],
            },
            int(shard["shard_index"]),
            tuple(
                TokenStreamIndex(**{
                    **asdict(row),
                    "shard_index": int(source_shard_index),
                    "token_offset": int(source["source_token_offset"]),
                })
                for row, source in zip(
                    selected_rows,
                    shard["source_coordinates"],
                )
            ),
        )
        chunk = cache_root / str(shard["chunk"])
        chunk_manifest = _validate_chunk(chunk, expected, verify_arrays=verify_arrays)
        if chunk_manifest is None:
            raise ValueError(f"panel stream cache chunk is invalid: {chunk.name}")
        if (
            shard["tokens_sha256"] != chunk_manifest["output_sha256"]["tokens.npy"]
            or shard["base_lengths_sha256"]
            != chunk_manifest["output_sha256"]["base_lengths.npy"]
            or int(shard["token_count"]) != int(chunk_manifest["tokens"])
        ):
            raise ValueError(f"panel stream cache shard manifest changed: {chunk.name}")
    return manifest


def build_panel_stream_cache(
    source_dataset: str | Path,
    output_dir: str | Path,
    *,
    stream_ids: Iterable[str],
    source_manifest_path: str | Path | None = None,
    source_index_path: str | Path | None = None,
    scratch_dir: str | Path | None = None,
    chunk_completed: Callable[[int, str], None] | None = None,
) -> dict[str, object]:
    """Build or resume an exact compact dataset, opening one source shard at a time.

    Completed source-shard chunks are immutable and independently checksummed.
    Publication uses sibling ``.partial`` files followed by atomic replacement,
    so an interrupted copy can invalidate only the current chunk.
    """

    source_root = Path(source_dataset)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(source_manifest_path) if source_manifest_path else source_root / "token_stream_manifest.json"
    source_manifest_bytes = manifest_path.read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    index_path = Path(source_index_path) if source_index_path else source_root / str(source_manifest["index"])
    if hashlib.sha256(index_path.read_bytes()).hexdigest() != source_manifest["index_sha256"]:
        raise ValueError("source token stream index checksum mismatch")
    selected = frozenset(map(str, stream_ids))
    if not selected:
        raise ValueError("panel stream cache requires selected stream IDs")
    source_rows = _read_index(index_path)
    lookup = {row.stream_id: row for row in source_rows}
    missing = sorted(selected - set(lookup))
    if missing:
        raise ValueError(f"selected streams are absent from source index: {missing[:5]}")
    contract = _cache_contract(hashlib.sha256(source_manifest_bytes).hexdigest(), source_manifest, selected)
    contract_hash = _hash(contract)
    existing_contract = output / "cache_contract.json"
    if existing_contract.is_file() and json.loads(existing_contract.read_text()) != contract:
        raise ValueError("existing panel stream cache has a different immutable contract")
    _atomic_json(existing_contract, contract)

    by_source: dict[int, list[TokenStreamIndex]] = {}
    for stream_id in selected:
        row = lookup[stream_id]
        by_source.setdefault(row.shard_index, []).append(row)
    for rows in by_source.values():
        rows.sort(key=lambda row: (row.token_offset, row.stream_id))
    source_shards = {int(row["shard_index"]): row for row in source_manifest["shards"]}
    if set(by_source) - set(source_shards):
        raise ValueError("source index refers to an absent shard")

    scratch_root = Path(scratch_dir) if scratch_dir else output / ".scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    final_rows: list[TokenStreamIndex] = []
    final_shards: list[dict[str, object]] = []
    for compact_index, source_index in enumerate(sorted(by_source)):
        source_shard = source_shards[source_index]
        rows = by_source[source_index]
        chunk_contract = _chunk_contract(contract_hash, source_shard, compact_index, rows)
        relative_chunk = Path("chunks") / f"source_shard_{source_index:05d}"
        chunk = output / relative_chunk
        if chunk.is_dir():
            for partial in chunk.glob("*.partial"):
                partial.unlink(missing_ok=True)
        valid = _validate_chunk(chunk, chunk_contract)
        if valid is None:
            work = scratch_root / f"source_shard_{source_index:05d}.partial"
            if work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True)
            tokens_source = np.load(source_root / str(source_shard["tokens"]), mmap_mode="r", allow_pickle=False)
            lengths_source = np.load(source_root / str(source_shard["base_lengths"]), mmap_mode="r", allow_pickle=False)
            token_count = sum(row.token_count for row in rows)
            tokens = np.empty(token_count, dtype=np.int32)
            lengths = np.empty(token_count, dtype=np.uint16)
            remapped: list[TokenStreamIndex] = []
            offset = 0
            source_token_digest = hashlib.sha256()
            source_length_digest = hashlib.sha256()
            for row in rows:
                end = row.token_offset + row.token_count
                token_range = np.asarray(tokens_source[row.token_offset:end], dtype=np.int32)
                length_range = np.asarray(lengths_source[row.token_offset:end], dtype=np.uint16)
                tokens[offset:offset + row.token_count] = token_range
                lengths[offset:offset + row.token_count] = length_range
                source_token_digest.update(token_range.tobytes(order="C"))
                source_length_digest.update(length_range.tobytes(order="C"))
                remapped.append(TokenStreamIndex(**{
                    **asdict(row), "shard_index": compact_index, "token_offset": offset,
                }))
                offset += row.token_count
            del tokens_source, lengths_source
            token_name, length_name, fragment_name = "tokens.npy", "base_lengths.npy", "index.jsonl"
            np.save(work / token_name, tokens, allow_pickle=False)
            np.save(work / length_name, lengths, allow_pickle=False)
            (work / fragment_name).write_text(
                "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in remapped),
                encoding="utf-8",
            )
            chunk_manifest = {
                "contract": chunk_contract,
                "streams": len(remapped),
                "tokens": token_count,
                "bases": sum(row.base_count for row in rows),
                "source_range_tokens_sha256": source_token_digest.hexdigest(),
                "source_range_base_lengths_sha256": source_length_digest.hexdigest(),
                "output_sha256": {
                    token_name: _sha256_file(work / token_name),
                    length_name: _sha256_file(work / length_name),
                    fragment_name: _sha256_file(work / fragment_name),
                },
            }
            _atomic_json(work / "chunk_manifest.json", chunk_manifest)
            _atomic_json(work / "COMPLETE.json", {
                "chunk_manifest_sha256": _sha256_file(work / "chunk_manifest.json")
            })
            chunk.mkdir(parents=True, exist_ok=True)
            (chunk / "COMPLETE.json").unlink(missing_ok=True)
            try:
                publication_order = [
                    work / name for name in (
                        "tokens.npy", "base_lengths.npy", "index.jsonl",
                        "chunk_manifest.json", "COMPLETE.json",
                    )
                ]
                for path in publication_order:
                    partial = chunk / (path.name + ".partial")
                    shutil.copyfile(path, partial)
                    os.replace(partial, chunk / path.name)
            except BaseException:
                for partial in chunk.glob("*.partial"):
                    partial.unlink(missing_ok=True)
                shutil.rmtree(work, ignore_errors=True)
                raise
            valid = _validate_chunk(chunk, chunk_contract)
            if valid is None:
                raise RuntimeError(f"published cache chunk failed validation: {chunk}")
            shutil.rmtree(work)
        fragment_rows = _read_index(chunk / "index.jsonl")
        final_rows.extend(fragment_rows)
        final_shards.append({
            "shard_index": compact_index,
            "tokens": str(relative_chunk / "tokens.npy"),
            "base_lengths": str(relative_chunk / "base_lengths.npy"),
            "token_count": int(valid["tokens"]),
            "tokens_sha256": valid["output_sha256"]["tokens.npy"],
            "base_lengths_sha256": valid["output_sha256"]["base_lengths.npy"],
            "chunk": str(relative_chunk),
            "source_shard_index": source_index,
            "source_tokens_sha256": source_shard["tokens_sha256"],
            "source_base_lengths_sha256": source_shard["base_lengths_sha256"],
            "source_coordinates": chunk_contract["source_coordinates"],
        })
        if chunk_completed is not None:
            chunk_completed(source_index, contract_hash)

    final_rows.sort(key=lambda row: (row.shard_index, row.token_offset, row.stream_id))
    final_index = output / "token_stream_index.jsonl"
    index_partial = final_index.with_name(final_index.name + ".partial")
    index_partial.write_text(
        "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in final_rows),
        encoding="utf-8",
    )
    os.replace(index_partial, final_index)
    compact_manifest = {
        "format_version": 1,
        "layout": "contig_indexed_lazy_segments",
        "segment_length": source_manifest["segment_length"],
        "streams": len(final_rows),
        "tokens": sum(row.token_count for row in final_rows),
        "bases": sum(row.base_count for row in final_rows),
        "index": final_index.name,
        "index_sha256": _sha256_file(final_index),
        "tokenizer": source_manifest["tokenizer"],
        "shards": final_shards,
        "parent_dataset_fingerprint": contract["parent_dataset_fingerprint"],
        "cache_contract_sha256": contract_hash,
        "source_shard_indices": sorted(by_source),
        "provenance": {
            "kind": "panel_stream_cache_v1",
            "source_manifest_sha256": contract["parent_dataset_fingerprint"],
        },
    }
    _atomic_json(output / "token_stream_manifest.json", compact_manifest)
    _atomic_json(output / "COMPLETE.json", {
        "manifest_sha256": _sha256_file(output / "token_stream_manifest.json"),
        "cache_contract_sha256": contract_hash,
    })
    return validate_panel_stream_cache(output)
