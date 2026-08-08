"""Contracts and statistics for Stage C context-anomaly and needle evaluation.

The functions in this module deliberately do not load a model.  This keeps case
selection, calibration, artifact identity, and checkpoint staging independently
auditable and cheap to unit test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from seqtrainer.data.bacteria_titan import StreamSegment


BENCHMARK_VERSION = "context_anomaly_v1"
DEFAULT_SEED = 20260807
SEGMENT_TOKENS = 32
INTERVENTIONS = ("carried", "reset", "wrong_host", "no_memory")
SEQUENCE_CLASSES = ("different_ani", "same_host", "untouched")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_hash(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def gc_fraction(sequence: str) -> float:
    sequence = str(sequence).upper()
    informative = sum(sequence.count(base) for base in "ACGT")
    return (sequence.count("G") + sequence.count("C")) / informative if informative else 0.0


@dataclass(frozen=True)
class ContextEvalConfig:
    split: str = "val"
    seed: int = DEFAULT_SEED
    mode: str = "smoke"
    hosts: int = 2
    insertion_segments: tuple[int, ...] = (1, 4)
    needle_distances: tuple[int, ...] = (3, 16)
    distractor_counts: tuple[int, ...] = (0,)
    warmup_segments: int = 16
    recovery_segments: int = 8
    gc_tolerance: float = 0.03
    bootstrap_samples: int = 2000
    benchmark_version: str = BENCHMARK_VERSION

    @classmethod
    def full(cls, *, split: str = "val", seed: int = DEFAULT_SEED) -> "ContextEvalConfig":
        return cls(
            split=split,
            seed=seed,
            mode="full",
            hosts=8,
            insertion_segments=(1, 2, 4, 8),
            needle_distances=(3, 8, 16, 32, 64),
            distractor_counts=(0, 4, 16),
        )

    def __post_init__(self) -> None:
        if self.split not in {"val", "test"}:
            raise ValueError("context evaluation split must be val or test")
        if self.mode not in {"smoke", "full"}:
            raise ValueError("context evaluation mode must be smoke or full")
        if self.hosts <= 0 or self.warmup_segments < 16 or self.recovery_segments < 8:
            raise ValueError("invalid host count, warm-up, or recovery requirement")
        if not self.insertion_segments or any(value <= 0 for value in self.insertion_segments):
            raise ValueError("insertion lengths must be positive")
        if not self.needle_distances or any(value <= 0 for value in self.needle_distances):
            raise ValueError("needle distances must be positive")
        if not 0 <= self.gc_tolerance <= 1 or self.bootstrap_samples <= 0:
            raise ValueError("invalid GC tolerance or bootstrap sample count")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("insertion_segments", "needle_distances", "distractor_counts"):
            payload[key] = list(payload[key])
        return payload


@dataclass(frozen=True)
class TokenStreamSlice:
    stream_id: str
    accession: str
    clade_group: str
    token_ids: tuple[int, ...]
    base_lengths: tuple[int, ...]
    dna: str

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.base_lengths):
            raise ValueError("token IDs and base lengths must align")
        if not self.stream_id or not self.accession or not self.clade_group:
            raise ValueError("stream identity is incomplete")

    @property
    def complete_segments(self) -> int:
        return max(0, (len(self.token_ids) - 1) // SEGMENT_TOKENS)

    def token_block(self, start_segment: int, length_segments: int) -> tuple[int, ...]:
        start = start_segment * SEGMENT_TOKENS
        end = start + length_segments * SEGMENT_TOKENS
        return self.token_ids[start:end]

    def base_block(self, start_segment: int, length_segments: int) -> str:
        start_token = start_segment * SEGMENT_TOKENS
        end_token = start_token + length_segments * SEGMENT_TOKENS
        start = sum(self.base_lengths[:start_token])
        end = sum(self.base_lengths[:end_token])
        return self.dna[start:end]


def decode_tokens_from_slice(stream: TokenStreamSlice, tokens: Sequence[int]) -> str:
    """Decode token IDs using the lossless token pieces retained by one stream."""

    pieces: dict[int, str] = {}
    cursor = 0
    for token, width in zip(stream.token_ids, stream.base_lengths):
        piece = stream.dna[cursor : cursor + width]
        cursor += width
        prior = pieces.setdefault(int(token), piece)
        if prior != piece:
            raise ValueError("one token ID maps to multiple DNA strings in a retained stream")
    missing = sorted(set(map(int, tokens)) - set(pieces))
    if missing:
        raise ValueError(f"retained stream cannot decode token IDs: {missing[:5]}")
    return "".join(pieces[int(token)] for token in tokens)


@dataclass(frozen=True)
class AnomalyCase:
    case_id: str
    host_stream_id: str
    host_accession: str
    host_clade_group: str
    donor_stream_id: str
    donor_accession: str
    donor_clade_group: str
    wrong_host_stream_id: str
    insertion_segment: int
    length_segments: int
    donor_start_segment: int
    hard_negative_start_segment: int
    host_window_start_segment: int
    host_window_end_segment: int
    donor_gc: float
    hard_negative_gc: float
    host_replaced_gc: float
    sequence_sha256: Mapping[str, str]
    retained_slice_sha256: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NeedleCase:
    case_id: str
    host_stream_id: str
    host_accession: str
    host_clade_group: str
    wrong_host_stream_id: str
    write_segment: int
    query_segment: int
    distance_segments: int
    distractor_count: int
    key_tokens: tuple[int, int, int, int]
    value_tokens: tuple[int, int]
    sequence_sha256: str
    retained_slice_sha256: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["key_tokens"] = list(self.key_tokens)
        payload["value_tokens"] = list(self.value_tokens)
        return payload


def stream_slice_from_segments(
    segments: Sequence[StreamSegment], decode: Callable[[Sequence[int]], str]
) -> TokenStreamSlice:
    if not segments:
        raise ValueError("cannot construct a slice from no segments")
    ordered = sorted(segments, key=lambda item: item.segment_index)
    if [item.segment_index for item in ordered] != list(range(len(ordered))):
        raise ValueError("stream segments must be a complete ordered prefix")
    first = ordered[0]
    tokens: list[int] = []
    lengths: list[int] = []
    for segment in ordered:
        count = sum(segment.valid_mask)
        tokens.extend(segment.input_ids[:count])
        # Input-token base lengths are the previous segment's label lengths.  The
        # first input is only needed for alignment, so infer its width from DNA.
        label_lengths = list(segment.represented_base_counts[:count])
        lengths.extend(([label_lengths[0] if label_lengths else 1] + label_lengths[:-1]))
    if ordered[-1].valid_tokens:
        tokens.append(ordered[-1].labels[ordered[-1].valid_tokens - 1])
        lengths.append(ordered[-1].represented_base_counts[ordered[-1].valid_tokens - 1])
    dna = decode(tokens)
    if sum(lengths) != len(dna):
        # Decoder truth is authoritative for known fixed tokenizers.
        pieces = [decode((token,)) for token in tokens]
        lengths = [len(piece) for piece in pieces]
    return TokenStreamSlice(
        first.stream_id,
        first.accession,
        first.clade_group,
        tuple(tokens),
        tuple(lengths),
        dna,
    )


def _ordered_streams(streams: Sequence[TokenStreamSlice], seed: int) -> list[TokenStreamSlice]:
    return sorted(streams, key=lambda item: sha256_bytes(f"{seed}:{item.stream_id}".encode()))


def _find_gc_block(
    stream: TokenStreamSlice,
    length: int,
    target_gc: float,
    tolerance: float,
    *,
    excluded: tuple[int, int] | None = None,
) -> tuple[int, float] | None:
    candidates: list[tuple[float, int, float]] = []
    for start in range(stream.complete_segments - length + 1):
        if excluded and not (start + length <= excluded[0] or start >= excluded[1]):
            continue
        value = gc_fraction(stream.base_block(start, length))
        difference = abs(value - target_gc)
        if difference <= tolerance + 1e-12:
            candidates.append((difference, start, value))
    if not candidates:
        return None
    _, start, value = min(candidates)
    return start, value


def materialize_anomaly_sequences(
    case: AnomalyCase, streams: Mapping[str, TokenStreamSlice]
) -> dict[str, tuple[int, ...]]:
    host = streams[case.host_stream_id]
    donor = streams[case.donor_stream_id]
    start = case.host_window_start_segment * SEGMENT_TOKENS
    end = case.host_window_end_segment * SEGMENT_TOKENS + 1
    original = list(host.token_ids[start:end])
    relative = (case.insertion_segment - case.host_window_start_segment) * SEGMENT_TOKENS
    width = case.length_segments * SEGMENT_TOKENS
    different = list(original)
    different[relative : relative + width] = donor.token_block(
        case.donor_start_segment, case.length_segments
    )
    hard = list(original)
    hard[relative : relative + width] = host.token_block(
        case.hard_negative_start_segment, case.length_segments
    )
    result = {
        "different_ani": tuple(different),
        "same_host": tuple(hard),
        "untouched": tuple(original),
    }
    for name, values in result.items():
        if sha256_bytes(np.asarray(values, dtype=np.int64).tobytes()) != case.sequence_sha256[name]:
            raise ValueError(f"anomaly sequence identity changed for {case.case_id}/{name}")
    return result


def select_anomaly_cases(
    streams: Sequence[TokenStreamSlice], config: ContextEvalConfig
) -> tuple[AnomalyCase, ...]:
    """Select deterministic, segment-aligned cross-group cases without relaxation."""

    ordered = _ordered_streams(streams, config.seed)
    longest = max(config.insertion_segments)
    needed = config.warmup_segments + longest + config.recovery_segments
    eligible_hosts = [item for item in ordered if item.complete_segments >= needed]
    cases: list[AnomalyCase] = []
    used_hosts: set[str] = set()
    for host in eligible_hosts:
        if host.stream_id in used_hosts:
            continue
        boundary = config.warmup_segments
        host_gc = gc_fraction(host.base_block(boundary, longest))
        donor_choice: tuple[TokenStreamSlice, int, float] | None = None
        for donor in ordered:
            if donor.accession == host.accession or donor.clade_group == host.clade_group:
                continue
            match = _find_gc_block(donor, longest, host_gc, config.gc_tolerance)
            if match is not None:
                donor_choice = donor, match[0], match[1]
                break
        if donor_choice is None:
            continue
        hard = _find_gc_block(
            host,
            longest,
            donor_choice[2],
            config.gc_tolerance,
            excluded=(boundary, boundary + longest),
        )
        if hard is None:
            continue
        wrong = next(
            (
                item
                for item in ordered
                if item.stream_id not in {host.stream_id, donor_choice[0].stream_id}
                and item.accession not in {host.accession, donor_choice[0].accession}
                and item.complete_segments >= config.warmup_segments
            ),
            donor_choice[0],
        )
        used_hosts.add(host.stream_id)
        donor, donor_start, _ = donor_choice
        for length in config.insertion_segments:
            donor_gc = gc_fraction(donor.base_block(donor_start, length))
            host_length_gc = gc_fraction(host.base_block(boundary, length))
            if abs(donor_gc - host_length_gc) > config.gc_tolerance + 1e-12:
                break
            hard_length = _find_gc_block(
                host,
                length,
                donor_gc,
                config.gc_tolerance,
                excluded=(boundary, boundary + length),
            )
            if hard_length is None:
                break
            hard_start, hard_gc = hard_length
            window_end = boundary + length + config.recovery_segments
            base = list(host.token_ids[: window_end * SEGMENT_TOKENS + 1])
            width = length * SEGMENT_TOKENS
            different = list(base)
            different[boundary * SEGMENT_TOKENS : boundary * SEGMENT_TOKENS + width] = donor.token_block(donor_start, length)
            same = list(base)
            same[boundary * SEGMENT_TOKENS : boundary * SEGMENT_TOKENS + width] = host.token_block(hard_start, length)
            hashes = {
                name: sha256_bytes(np.asarray(value, dtype=np.int64).tobytes())
                for name, value in (("different_ani", different), ("same_host", same), ("untouched", base))
            }
            identity = f"{host.stream_id}|{donor.stream_id}|{length}|{config.seed}"
            cases.append(
                AnomalyCase(
                    case_id="anomaly_" + sha256_bytes(identity.encode())[:16],
                    host_stream_id=host.stream_id,
                    host_accession=host.accession,
                    host_clade_group=host.clade_group,
                    donor_stream_id=donor.stream_id,
                    donor_accession=donor.accession,
                    donor_clade_group=donor.clade_group,
                    wrong_host_stream_id=wrong.stream_id,
                    insertion_segment=boundary,
                    length_segments=length,
                    donor_start_segment=donor_start,
                    hard_negative_start_segment=hard_start,
                    host_window_start_segment=0,
                    host_window_end_segment=window_end,
                    donor_gc=donor_gc,
                    hard_negative_gc=hard_gc,
                    host_replaced_gc=host_length_gc,
                    sequence_sha256=hashes,
                    retained_slice_sha256={
                        "host_window": sha256_bytes(host.base_block(0, window_end).encode("ascii")),
                        "donor_fragment": sha256_bytes(donor.base_block(donor_start, length).encode("ascii")),
                        "hard_negative": sha256_bytes(host.base_block(hard_start, length).encode("ascii")),
                        "wrong_host_warmup": sha256_bytes(wrong.base_block(0, config.warmup_segments).encode("ascii")),
                    },
                )
            )
        if len(used_hosts) == config.hosts:
            break
    if len(used_hosts) != config.hosts or len(cases) != config.hosts * len(config.insertion_segments):
        raise ValueError(
            f"could not construct {config.hosts} cross-ANI host/donor pairs at GC tolerance "
            f"{config.gc_tolerance:.3f}; constructed {len(used_hosts)}"
        )
    return tuple(cases)


def association_occurrences(tokens: Sequence[int], key: Sequence[int], value: Sequence[int]) -> list[int]:
    pattern = tuple(key) + tuple(value)
    return [
        index
        for index in range(len(tokens) - len(pattern) + 1)
        if tuple(tokens[index : index + len(pattern)]) == pattern
    ]


def select_needle_cases(
    streams: Sequence[TokenStreamSlice], config: ContextEvalConfig
) -> tuple[NeedleCase, ...]:
    """Build natural-token key/value cases whose association is unique."""

    ordered = _ordered_streams(streams, config.seed + 1)
    cases: list[NeedleCase] = []
    selected_hosts = 0
    for host in ordered:
        required = config.warmup_segments + max(config.needle_distances) + 2
        if host.complete_segments < required:
            continue
        write = config.warmup_segments * SEGMENT_TOKENS
        key = tuple(host.token_ids[write : write + 4])
        value = tuple(host.token_ids[write + 4 : write + 6])
        if len(key) != 4 or len(value) != 2 or len(set(value)) == 0:
            continue
        if association_occurrences(host.token_ids, key, value) != [write]:
            continue
        wrong = next(
            (
                item for item in ordered
                if item.accession != host.accession and item.complete_segments >= required
            ),
            None,
        )
        if wrong is None:
            continue
        valid: list[NeedleCase] = []
        for distance in config.needle_distances:
            # Distance counts complete segments *between* the write and query.
            query_segment = config.warmup_segments + 1 + distance
            for distractors in config.distractor_counts:
                sequence = materialize_needle_sequence_values(
                    host, query_segment=query_segment, key=key, value=value,
                    distractor_count=distractors,
                )
                identity = f"{host.stream_id}|{distance}|{distractors}|{config.seed}"
                valid.append(
                    NeedleCase(
                        case_id="needle_" + sha256_bytes(identity.encode())[:16],
                        host_stream_id=host.stream_id,
                        host_accession=host.accession,
                        host_clade_group=host.clade_group,
                        wrong_host_stream_id=wrong.stream_id,
                        write_segment=config.warmup_segments,
                        query_segment=query_segment,
                        distance_segments=distance,
                        distractor_count=distractors,
                        key_tokens=key,  # type: ignore[arg-type]
                        value_tokens=value,  # type: ignore[arg-type]
                        sequence_sha256=sha256_bytes(np.asarray(sequence, dtype=np.int64).tobytes()),
                        retained_slice_sha256={
                            "needle_window": sha256_bytes(decode_tokens_from_slice(host, sequence).encode("ascii")),
                            "wrong_host_warmup": sha256_bytes(wrong.base_block(0, query_segment).encode("ascii")),
                        },
                    )
                )
        cases.extend(valid)
        selected_hosts += 1
        if selected_hosts == config.hosts:
            break
    expected = config.hosts * len(config.needle_distances) * len(config.distractor_counts)
    if selected_hosts != config.hosts or len(cases) != expected:
        raise ValueError(f"could not construct unique natural needle cases for {config.hosts} hosts")
    return tuple(cases)


def materialize_needle_sequence_values(
    host: TokenStreamSlice,
    *,
    query_segment: int,
    key: Sequence[int],
    value: Sequence[int],
    distractor_count: int,
) -> tuple[int, ...]:
    """Create one teacher-forced query with deterministic intervening associations."""

    end = (query_segment + 1) * SEGMENT_TOKENS + 1
    sequence = list(host.token_ids[:end])
    natural_writes = association_occurrences(host.token_ids, key, value)
    if len(natural_writes) != 1:
        raise ValueError("needle association is not unique in the natural host")
    query = query_segment * SEGMENT_TOKENS
    sequence[query : query + 4] = key
    sequence[query + 4 : query + 6] = value
    if distractor_count:
        start = (natural_writes[0] // SEGMENT_TOKENS + 1) * SEGMENT_TOKENS
        available = max(query - start, 0)
        spacing = max(6, available // distractor_count)
        for index in range(distractor_count):
            destination = start + index * spacing
            source = 8 * SEGMENT_TOKENS + index * 6
            if destination + 6 > query or source + 6 > len(host.token_ids):
                raise ValueError("needle window cannot accommodate requested distractors")
            association = list(host.token_ids[source : source + 6])
            if tuple(association[:4]) == tuple(key):
                association[0] = next(
                    int(token) for token in host.token_ids if int(token) != int(key[0])
                )
            sequence[destination : destination + 6] = association
    expected = {natural_writes[0], query}
    for _ in range(8):
        extras = set(association_occurrences(sequence, key, value)) - expected
        if not extras:
            break
        for start in extras:
            sequence[start] = next(
                int(token) for token in host.token_ids if int(token) != int(key[0])
            )
    if set(association_occurrences(sequence, key, value)) != expected:
        raise ValueError("could not preserve a unique declared needle write/query association")
    return tuple(sequence)


def materialize_needle_sequence(
    case: NeedleCase, streams: Mapping[str, TokenStreamSlice]
) -> tuple[int, ...]:
    values = materialize_needle_sequence_values(
        streams[case.host_stream_id],
        query_segment=case.query_segment,
        key=case.key_tokens,
        value=case.value_tokens,
        distractor_count=case.distractor_count,
    )
    if sha256_bytes(np.asarray(values, dtype=np.int64).tobytes()) != case.sequence_sha256:
        raise ValueError(f"needle sequence identity changed for {case.case_id}")
    return values


def bpb(nll_nats: float, represented_bases: int) -> float:
    if represented_bases <= 0:
        raise ValueError("BPB requires a positive base count")
    return float(nll_nats) / (math.log(2.0) * represented_bases)


def context_conflict(carried_bpb: float, reset_bpb: float) -> float:
    return float(carried_bpb) - float(reset_bpb)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(labels) != len(scores) or not labels:
        raise ValueError("AUPRC inputs must be non-empty and aligned")
    positives = sum(bool(value) for value in labels)
    if positives == 0:
        return 0.0
    ordered = sorted(range(len(labels)), key=lambda index: (-float(scores[index]), index))
    hits = 0
    total = 0.0
    for rank, index in enumerate(ordered, 1):
        if labels[index]:
            hits += 1
            total += hits / rank
    return total / positives


def threshold_at_fpr(negative_scores: Sequence[float], fpr: float) -> float:
    if not negative_scores or not 0 <= fpr <= 1:
        raise ValueError("fixed-FPR calibration requires negatives and FPR in [0,1]")
    values = sorted(map(float, negative_scores), reverse=True)
    allowed = int(math.floor(fpr * len(values)))
    return math.nextafter(values[allowed], math.inf) if allowed < len(values) else -math.inf


def detection_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    thresholds: Mapping[str, float] | None = None,
    represented_bases: Sequence[int] | None = None,
) -> dict[str, object]:
    if len(labels) != len(scores) or not labels:
        raise ValueError("detection arrays must be non-empty and aligned")
    negatives = [float(score) for label, score in zip(labels, scores) if not label]
    positives = [float(score) for label, score in zip(labels, scores) if label]
    if not negatives or not positives:
        raise ValueError("detection metrics require both classes")
    fixed = dict(thresholds or {
        "fpr_0.01": threshold_at_fpr(negatives, 0.01),
        "fpr_0.001": threshold_at_fpr(negatives, 0.001),
    })
    base_values = list(represented_bases or [1] * len(labels))
    if len(base_values) != len(labels):
        raise ValueError("represented base counts must align with detection arrays")
    negative_bases = sum(value for label, value in zip(labels, base_values) if not label)
    if negative_bases <= 0:
        raise ValueError("false-positive rate requires represented negative bases")
    result: dict[str, object] = {"auprc": average_precision(labels, scores), "thresholds": fixed}
    for name, threshold in fixed.items():
        true_positive = sum(score >= threshold for score in positives) / len(positives)
        false_positive = sum(score >= threshold for score in negatives)
        result[f"tpr_at_{name.removeprefix('fpr_')}"] = true_positive
        result[f"false_positives_per_mb_at_{name.removeprefix('fpr_')}"] = false_positive * 1_000_000 / negative_bases
    return result


def boundary_metrics(
    scores: Sequence[float], *, boundary: int, threshold: float, positive_end: int
) -> dict[str, int | None]:
    detections = [index for index, score in enumerate(scores) if score >= threshold]
    first = next((index for index in detections if index >= boundary), None)
    nearest = min(detections, key=lambda index: abs(index - boundary)) if detections else None
    return {
        "detection_delay": None if first is None else first - boundary,
        "boundary_error": None if nearest is None else nearest - boundary,
        "recovery_time": next(
            (index - positive_end for index in range(positive_end, len(scores)) if scores[index] < threshold),
            None,
        ),
    }


def paired_host_bootstrap(
    values: Mapping[str, Sequence[float] | float], *, seed: int = DEFAULT_SEED, samples: int = 2000
) -> dict[str, float | int]:
    hosts = sorted(values)
    if not hosts:
        raise ValueError("host bootstrap requires values")
    host_means = {
        host: float(np.mean(value if isinstance(value, Sequence) else [value]))
        for host, value in values.items()
    }
    rng = random.Random(seed)
    draws = sorted(
        sum(host_means[hosts[rng.randrange(len(hosts))]] for _ in hosts) / len(hosts)
        for _ in range(samples)
    )
    return {
        "hosts": len(hosts),
        "mean": sum(host_means.values()) / len(hosts),
        "lower_95": draws[int(0.025 * (samples - 1))],
        "upper_95": draws[int(0.975 * (samples - 1))],
    }


def needle_metrics(rows: Sequence[Mapping[str, object]], *, seed: int, samples: int) -> dict[str, object]:
    gains: dict[str, list[float]] = {}
    by_distance: dict[int, list[float]] = {}
    by_interference: dict[int, list[float]] = {}
    host_long_direction: dict[str, list[float]] = {}
    for row in rows:
        host = str(row["host_accession"])
        gain = float(row["carried_log_probability"])-float(row["reset_log_probability"])
        gains.setdefault(host, []).append(gain)
        distance = int(row["distance_segments"])
        by_distance.setdefault(distance, []).append(gain)
        by_interference.setdefault(int(row["distractor_count"]), []).append(gain)
        if distance >= 16:
            host_long_direction.setdefault(host, []).append(gain)
    long_values = {
        host: values for host, values in host_long_direction.items() if values
    }
    long_bootstrap = paired_host_bootstrap(long_values, seed=seed, samples=samples) if long_values else None
    positive_hosts = sum(float(np.mean(values)) > 0 for values in long_values.values())
    result: dict[str, object] = {
        "carried_minus_reset": paired_host_bootstrap(gains, seed=seed, samples=samples),
        "distance_decay": {str(key): float(np.mean(value)) for key, value in sorted(by_distance.items())},
        "interference_decay": {str(key): float(np.mean(value)) for key, value in sorted(by_interference.items())},
        "long_context_bootstrap": long_bootstrap,
        "long_context_positive_hosts": positive_hosts,
        "promising_long_context_signal": bool(
            long_bootstrap and float(long_bootstrap["lower_95"]) > 0 and positive_hosts >= 6
        ),
    }
    secondary = (
        "first_token_rank", "recall_at_1", "recall_at_5",
        "reciprocal_rank", "exact_two_token_recovery",
    )
    for key in secondary:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            result[key] = float(np.mean(values))
    return result


def _copy_verified(source: Path, partial: Path, *, copy: Callable[[Path, Path], object]) -> tuple[int, str]:
    before_size, before_hash = source.stat().st_size, sha256_file(source)
    copy(source, partial)
    destination_size, destination_hash = partial.stat().st_size, sha256_file(partial)
    after_size, after_hash = source.stat().st_size, sha256_file(source)
    if (before_size, before_hash) != (after_size, after_hash):
        partial.unlink(missing_ok=True)
        raise RuntimeError("source checkpoint changed during copying")
    if (before_size, before_hash) != (destination_size, destination_hash):
        partial.unlink(missing_ok=True)
        raise RuntimeError("checkpoint copy identity mismatch")
    return before_size, before_hash


def stage_checkpoint(
    source: str | Path,
    registry: str | Path,
    metadata: Mapping[str, object],
    *,
    trusted: bool,
    copy: Callable[[Path, Path], object] = shutil.copyfile,
) -> Path:
    """Copy a full-state checkpoint into an immutable, content-addressed model entry."""

    if not trusted:
        raise ValueError("full-state checkpoint staging requires an explicit trust declaration")
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    step = int(metadata.get("optimizer_step", -1))
    if step < 0:
        raise ValueError("checkpoint metadata requires optimizer_step")
    source_hash = sha256_file(source_path)
    model_id = f"{step}_{source_hash[:12]}"
    model_dir = Path(registry) / "models" / model_id
    checkpoint = model_dir / "checkpoint.pt"
    manifest_path = model_dir / "model_manifest.json"
    if model_dir.exists():
        if checkpoint.is_file() and sha256_file(checkpoint) == source_hash and manifest_path.is_file():
            return model_dir
        raise FileExistsError(f"immutable model entry already exists with different content: {model_dir}")
    model_dir.mkdir(parents=True)
    partial = model_dir / "checkpoint.pt.partial"
    try:
        size, digest = _copy_verified(source_path, partial, copy=copy)
        if digest != source_hash:
            partial.unlink(missing_ok=True)
            raise RuntimeError("source checkpoint changed before verified copying")
        os.replace(partial, checkpoint)
        manifest = {
            "format_version": 1,
            "model_id": model_id,
            "checkpoint_sha256": digest,
            "checkpoint_size": size,
            "source_path": str(source_path),
            "trusted_full_state": True,
            **dict(metadata),
        }
        _atomic_json(manifest_path, manifest)
    except BaseException:
        partial.unlink(missing_ok=True)
        if not any(model_dir.iterdir()):
            model_dir.rmdir()
        raise
    return model_dir


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


class CaseResumeStore:
    """Atomic per-case JSON resume store locked to one manifest contract."""

    def __init__(self, root: str | Path, contract: Mapping[str, object]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = dict(contract)
        self.hash = contract_hash(self.contract)
        contract_path = self.root / "contract.json"
        if contract_path.exists():
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
            if existing != {"contract": self.contract, "contract_sha256": self.hash}:
                raise ValueError("resume contract changed; refuse unsafe resume")
        else:
            _atomic_json(contract_path, {"contract": self.contract, "contract_sha256": self.hash})

    def completed(self, case_id: str) -> bool:
        return (self.root / f"{case_id}.json").is_file()

    def load(self, case_id: str) -> Mapping[str, object]:
        return json.loads((self.root / f"{case_id}.json").read_text(encoding="utf-8"))

    def commit(self, case_id: str, payload: Mapping[str, object]) -> Path:
        path = self.root / f"{case_id}.json"
        wrapped = {"case_id": case_id, "contract_sha256": self.hash, "result": dict(payload)}
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != wrapped:
                raise ValueError(f"completed case changed: {case_id}")
            return path
        _atomic_json(path, wrapped)
        return path


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
    os.replace(partial, destination)
    return destination


def write_sequence_slices(
    path: str | Path, records: Iterable[tuple[str, str]]
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with gzip.open(partial, "wt", encoding="ascii", newline="\n") as handle:
        for identity, sequence in records:
            handle.write(f">{identity}\n")
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset : offset + 80] + "\n")
    os.replace(partial, destination)
    return destination


def regenerate_catalog(registry: str | Path) -> Path:
    root = Path(registry)
    rows: list[dict[str, object]] = []
    for complete in sorted((root / "evaluations" / BENCHMARK_VERSION).glob("*/*/*/COMPLETE.json")):
        directory = complete.parent
        manifest_path = directory / "evaluation_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.append({
            "model_id": manifest.get("model_id"),
            "split": manifest.get("split"),
            "config_sha256": manifest.get("config_sha256"),
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "dataset_fingerprint": manifest.get("dataset_fingerprint"),
            "panel_sha256": manifest.get("panel_sha256"),
            "completed_at": json.loads(complete.read_text(encoding="utf-8")).get("completed_at"),
            "path": str(directory),
        })
    destination = root / "EVALUATION_CATALOG.csv"
    fields = ["model_id", "split", "config_sha256", "checkpoint_sha256", "dataset_fingerprint", "panel_sha256", "completed_at", "path"]
    partial = destination.with_suffix(".csv.partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, destination)
    return destination
