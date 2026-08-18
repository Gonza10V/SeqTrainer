"""Frozen scientific contract and analysis primitives for notebook 03q.

This module keeps panel identity, host-clustered inference, and nested model
selection outside the notebook so the expensive A100 run is reproducible and
the statistical implementation can be tested on a CPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .context_eval import TokenStreamSlice, gc_fraction


STUDY_VERSION = "c16_c19_anomaly_needle_v2"
C16_SHA256 = "21898362291f4fd1e6aafcfbe47e8b05dbe69e5c8036e6ae7927a6ac24ac4541"
C19_SHA256 = "07fb2069b1f29a76898a90d8dfb899c5ca46cb90608fac45bc0ddff9876dbd1a"
CHECKPOINT_SHA256 = {"C16": C16_SHA256, "C19": C19_SHA256}
SEGMENT_TOKENS = 32
SEGMENT_BASES = 192
ANOMALY_LENGTHS = (1, 4, 16, 64)
INSERTION_DEPTHS = (16, 128)
BOUNDED_ANOMALY_LENGTHS = (1, 16, 64)
BOUNDED_INSERTION_DEPTHS = (16,)
NEEDLE_DISTANCES = (3, 8, 16, 32, 64)
NEEDLE_DISTRACTORS = (0, 4, 16)
BOUNDED_NEEDLE_DISTANCES = (3, 16, 64)
BOUNDED_NEEDLE_DISTRACTORS = (0, 16)
ANOMALY_CLASSES = ("foreign_near", "foreign_far", "same_host", "untouched")
MEMORY_INTERVENTIONS = ("carried", "reset", "wrong_host", "no_memory")
NEEDLE_CONTROLS = (
    "carried", "reset", "wrong_host", "no_memory", "needle_absent", "incorrect_value"
)
REGULARIZATION_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

REFERENCE_FEATURES = ("bpb", "markov_nll", "tetranucleotide_distance")
MEMORY_FEATURES = REFERENCE_FEATURES + (
    "momentary_surprise_rms_max_log1p",
    "surprise_norm_log1p_diff",
    "memory_update_norm_log1p",
    "retrieval_norm_log1p",
    "state_drift_norm_log1p_diff",
    "theta_mean",
)
COUNTERFACTUAL_FEATURES = (
    "memory_detector_logit",
    "context_conflict",
    "carried_minus_reset_surprise_log1p",
)


def evaluation_interventions(mode: str, sequence_class: str) -> tuple[str, ...]:
    """Return the frozen intervention arms for one materialized sequence."""

    if mode == "bounded" and not sequence_class.startswith("foreign_"):
        return ("carried",)
    return MEMORY_INTERVENTIONS


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def contract_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ScientificStudyConfig:
    mode: str = "smoke"
    hosts: int = 2
    seed: int = 20260813
    lengths: tuple[int, ...] = (1, 4)
    depths: tuple[int, ...] = (16,)
    needle_distances: tuple[int, ...] = (3, 16)
    needle_distractors: tuple[int, ...] = (0,)
    native_calibration_segments: int = 128
    pre_segments: int = 8
    recovery_segments: int = 16
    gc_tolerance: float = 0.03
    ani_minimum: float = 95.0
    ani_stratum_gap: float = 1.5
    bootstrap_samples: int = 2000
    circular_permutations: int = 10_000

    @classmethod
    def full(cls, *, seed: int = 20260813) -> "ScientificStudyConfig":
        return cls(
            mode="full", hosts=8, seed=seed, lengths=ANOMALY_LENGTHS,
            depths=INSERTION_DEPTHS, needle_distances=NEEDLE_DISTANCES,
            needle_distractors=NEEDLE_DISTRACTORS,
        )

    @classmethod
    def bounded(cls, *, seed: int = 20260813) -> "ScientificStudyConfig":
        """Eight-host validation contract sized for one C19 A100 session."""

        return cls(
            mode="bounded", hosts=8, seed=seed,
            lengths=BOUNDED_ANOMALY_LENGTHS,
            depths=BOUNDED_INSERTION_DEPTHS,
            needle_distances=BOUNDED_NEEDLE_DISTANCES,
            needle_distractors=BOUNDED_NEEDLE_DISTRACTORS,
        )

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "bounded", "full"}:
            raise ValueError("study mode must be smoke, bounded, or full")
        if self.hosts != (2 if self.mode == "smoke" else 8):
            raise ValueError("smoke requires two hosts; bounded/full require eight hosts")
        if self.mode == "bounded" and (
            self.lengths != BOUNDED_ANOMALY_LENGTHS
            or self.depths != BOUNDED_INSERTION_DEPTHS
            or self.needle_distances != BOUNDED_NEEDLE_DISTANCES
            or self.needle_distractors != BOUNDED_NEEDLE_DISTRACTORS
        ):
            raise ValueError("bounded mode requires the exact sub-24-hour grid")
        if self.mode == "full" and (
            self.lengths != ANOMALY_LENGTHS or self.depths != INSERTION_DEPTHS
            or self.needle_distances != NEEDLE_DISTANCES
            or self.needle_distractors != NEEDLE_DISTRACTORS
        ):
            raise ValueError("full mode requires the exact preregistered anomaly and needle grids")
        if self.native_calibration_segments != 128:
            raise ValueError("the protocol reserves exactly 128 native calibration segments per host")
        if self.gc_tolerance != 0.03 or self.ani_minimum != 95.0:
            raise ValueError("GC and ANI thresholds are frozen by the scientific contract")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("lengths", "depths", "needle_distances", "needle_distractors"):
            payload[key] = list(payload[key])
        return payload


def planned_segment_forwards(config: ScientificStudyConfig) -> dict[str, int]:
    """Count model segment-forwards before either checkpoint is evaluated.

    Bounded mode evaluates all four memory interventions only on the foreign
    sequence and carried memory on each matched negative.  Smoke/full retain
    the older Cartesian intervention design.
    """

    donor_strata = 2
    cases = config.hosts * donor_strata * len(config.depths) * len(config.lengths)
    arms_per_case = 6 if config.mode == "bounded" else 12
    anomaly_scoring = sum(
        config.hosts * donor_strata * arms_per_case
        * (length + config.pre_segments + config.recovery_segments)
        for _depth in config.depths
        for length in config.lengths
    )
    anomaly_warmup = sum(
        config.hosts * donor_strata * len(config.lengths)
        * 2 * max(depth - config.pre_segments, 0)
        for depth in config.depths
    )
    native_calibration = config.hosts * config.native_calibration_segments
    # Needle write is fixed at segment 16 by the frozen ContextEvalConfig.
    needle = sum(
        config.hosts * len(config.needle_distractors)
        * (4 * (17 + distance) + len(NEEDLE_CONTROLS))
        for distance in config.needle_distances
    )
    components = {
        "anomaly_cases": cases,
        "anomaly_scoring": anomaly_scoring,
        "anomaly_warmup": anomaly_warmup,
        "native_calibration": native_calibration,
        "needle_scoring_and_warmup": needle,
    }
    components["total_segment_forwards"] = sum(
        value for key, value in components.items() if key != "anomaly_cases"
    )
    return components


def runtime_projection(
    *,
    planned_forwards: int,
    measured_segments_per_second: float,
    probe_forwards: int = 64,
    reference_segments_per_second: float = 0.474,
    contingency: float = 1.20,
    fixed_overhead_hours: float = 0.5,
    maximum_hours: float = 22.0,
) -> dict[str, float | int | bool]:
    """Conservative A100 projection using the slower measured/reference rate."""

    if planned_forwards <= 0 or probe_forwards <= 0:
        raise ValueError("runtime projection requires positive forward counts")
    if measured_segments_per_second <= 0 or reference_segments_per_second <= 0:
        raise ValueError("runtime projection requires positive throughput")
    effective_rate = min(measured_segments_per_second, reference_segments_per_second)
    raw_hours = (planned_forwards + probe_forwards) / effective_rate / 3600.0
    projected_hours = raw_hours * contingency + fixed_overhead_hours
    return {
        "planned_segment_forwards": planned_forwards,
        "probe_segment_forwards": probe_forwards,
        "measured_segments_per_second": measured_segments_per_second,
        "reference_segments_per_second": reference_segments_per_second,
        "effective_segments_per_second": effective_rate,
        "raw_hours": raw_hours,
        "contingency": contingency,
        "fixed_overhead_hours": fixed_overhead_hours,
        "projected_hours": projected_hours,
        "maximum_hours": maximum_hours,
        "accepted": projected_hours <= maximum_hours,
    }


@dataclass(frozen=True)
class DonorPair:
    host_accession: str
    near_accession: str
    near_ani: float
    far_accession: str
    far_ani: float
    host_ani99_group: str
    near_ani99_group: str
    far_ani99_group: str

    def __post_init__(self) -> None:
        if len({self.host_accession, self.near_accession, self.far_accession}) != 3:
            raise ValueError("host, near donor, and far donor must be distinct accessions")
        if len({self.host_ani99_group, self.near_ani99_group, self.far_ani99_group}) != 3:
            raise ValueError("host, near donor, and far donor must have distinct ANI99 groups")
        if min(self.near_ani, self.far_ani) < 95.0:
            raise ValueError("donor ANI must be at least 95%")
        if self.near_ani - self.far_ani < 1.5:
            raise ValueError("near/far strata must differ by at least 1.5 ANI percentage points")


def _pair_lookup(ani: pd.DataFrame) -> dict[tuple[str, str], float]:
    aliases = ({"Ref_file": "left", "Query_file": "right", "ANI": "ani"}
               if {"Ref_file", "Query_file", "ANI"}.issubset(ani.columns) else {})
    frame = ani.rename(columns=aliases)
    if not {"left", "right", "ani"}.issubset(frame.columns):
        raise ValueError("ANI evidence requires left, right, and ani columns")
    result: dict[tuple[str, str], float] = {}
    accession_pattern = re.compile(r"(?:GCF|GCA)_\d+\.\d+")
    for row in frame[["left", "right", "ani"]].itertuples(index=False):
        left_match, right_match = accession_pattern.search(str(row.left)), accession_pattern.search(str(row.right))
        left = left_match.group(0) if left_match else Path(str(row.left)).stem
        right = right_match.group(0) if right_match else Path(str(row.right)).stem
        result[tuple(sorted((left, right)))] = float(row.ani)
    return result


def choose_relative_donors(
    hosts: Sequence[str],
    e25_donors: Sequence[str],
    ani_evidence: pd.DataFrame,
    ani99_groups: Mapping[str, str],
) -> tuple[DonorPair, ...]:
    """Choose exact nearest/farthest eligible E25 donors for each host."""

    lookup = _pair_lookup(ani_evidence)
    donor_set = sorted(set(map(str, e25_donors)))
    pairs: list[DonorPair] = []
    for host in sorted(set(map(str, hosts))):
        if host not in ani99_groups:
            raise ValueError(f"host {host} has no ANI99 group")
        candidates: list[tuple[float, str]] = []
        for donor in donor_set:
            if donor == host or donor not in ani99_groups:
                continue
            if ani99_groups[donor] == ani99_groups[host]:
                continue
            value = lookup.get(tuple(sorted((host, donor))))
            if value is not None and value >= 95.0:
                candidates.append((value, donor))
        if len(candidates) < 2:
            raise ValueError(f"host {host} lacks two eligible E25 donors with ANI >=95%")
        candidates.sort(key=lambda item: (item[0], item[1]))
        eligible_pairs = [
            (near_ani - far_ani, near_ani, -far_ani, near, far)
            for far_ani, far in candidates
            for near_ani, near in candidates
            if near_ani - far_ani >= 1.5
            and ani99_groups[near] != ani99_groups[far]
        ]
        if not eligible_pairs:
            raise ValueError(f"host {host} lacks distinct near/far ANI99 donor strata")
        _, near_ani, negative_far_ani, near, far = max(eligible_pairs)
        far_ani = -negative_far_ani
        pairs.append(DonorPair(
            host, near, near_ani, far, far_ani, ani99_groups[host],
            ani99_groups[near], ani99_groups[far],
        ))
    return tuple(pairs)


@dataclass(frozen=True)
class FrozenAnomalyCase:
    case_id: str
    host_stream_id: str
    host_accession: str
    donor_stream_id: str
    donor_accession: str
    donor_distance: str
    donor_ani: float
    insertion_depth: int
    length_segments: int
    donor_start_segment: int
    same_host_start_segment: int
    native_calibration_start_segment: int
    donor_gc: float
    replaced_host_gc: float
    same_host_gc: float
    token_sha256: Mapping[str, str]
    dna_sha256: Mapping[str, str]
    source_coordinate_sha256: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _token_hash(tokens: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def _coordinate_hash(stream: TokenStreamSlice, start: int, length: int) -> str:
    payload = {
        "stream_id": stream.stream_id,
        "accession": stream.accession,
        "start_segment": start,
        "length_segments": length,
        "tokens_sha256": _token_hash(stream.token_block(start, length)),
        "dna_sha256": hashlib.sha256(stream.base_block(start, length).encode("ascii")).hexdigest(),
    }
    return contract_hash(payload)


def _token_dna(stream: TokenStreamSlice, start_token: int, end_token: int) -> str:
    left = sum(stream.base_lengths[:start_token])
    right = sum(stream.base_lengths[:end_token])
    return stream.dna[left:right]


def _canonical_dna(value: str) -> bool:
    return bool(value) and set(value.upper()) <= set("ACGT")


def canonical_host_calibration_start(
    stream: TokenStreamSlice, config: ScientificStudyConfig
) -> int | None:
    """Return the earliest valid calibration window for a retained host stream.

    The anomaly and longest needle both retain a prefix from token zero.  The
    calibration window must be canonical, include the one-token scoring tail,
    and not overlap that retained prefix.
    """

    maximum_end = max(config.depths) + max(config.lengths) + config.recovery_segments
    calibration_segments = config.native_calibration_segments
    if stream.complete_segments < maximum_end + calibration_segments:
        return None
    prefix_end_token = maximum_end * SEGMENT_TOKENS + 1
    if not _canonical_dna(_token_dna(stream, 0, prefix_end_token)):
        return None
    final_start = stream.complete_segments - calibration_segments
    for start in range(maximum_end, final_start + 1):
        end_token = (start + calibration_segments) * SEGMENT_TOKENS + 1
        if _canonical_dna(_token_dna(stream, start * SEGMENT_TOKENS, end_token)):
            return start
    return None


def _best_gc_block(
    stream: TokenStreamSlice,
    length: int,
    target_gc: float,
    *,
    tolerance: float,
    excluded: Sequence[tuple[int, int]] = (),
) -> tuple[int, float] | None:
    candidates: list[tuple[float, int, float]] = []
    for start in range(stream.complete_segments - length + 1):
        if any(not (start + length <= left or start >= right) for left, right in excluded):
            continue
        dna = stream.base_block(start, length)
        if not _canonical_dna(dna):
            continue
        value = gc_fraction(dna)
        if abs(value - target_gc) <= tolerance + 1e-12:
            candidates.append((abs(value - target_gc), start, value))
    if not candidates:
        return None
    _, start, value = min(candidates)
    return start, value


def materialize_frozen_case(
    case: FrozenAnomalyCase, streams: Mapping[str, TokenStreamSlice], *, recovery_segments: int = 16
) -> dict[str, tuple[int, ...]]:
    host, donor = streams[case.host_stream_id], streams[case.donor_stream_id]
    end = case.insertion_depth + case.length_segments + recovery_segments
    untouched = list(host.token_ids[: end * SEGMENT_TOKENS + 1])
    left = case.insertion_depth * SEGMENT_TOKENS
    width = case.length_segments * SEGMENT_TOKENS
    foreign = untouched.copy()
    foreign[left:left + width] = donor.token_block(case.donor_start_segment, case.length_segments)
    same_host = untouched.copy()
    same_host[left:left + width] = host.token_block(case.same_host_start_segment, case.length_segments)
    result = {
        f"foreign_{case.donor_distance}": tuple(foreign),
        "same_host": tuple(same_host),
        "untouched": tuple(untouched),
    }
    if any(len(tokens) != len(untouched) for tokens in result.values()):
        raise ValueError("replacement changed token length")
    for label, tokens in result.items():
        if _token_hash(tokens) != case.token_sha256[label]:
            raise ValueError(f"frozen token array changed for {case.case_id}/{label}")
    return result


def freeze_anomaly_panel(
    host_streams: Sequence[TokenStreamSlice],
    donor_streams: Sequence[TokenStreamSlice],
    donor_pairs: Sequence[DonorPair],
    config: ScientificStudyConfig,
) -> tuple[tuple[FrozenAnomalyCase, ...], dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    """Materialize the entire replacement panel before either checkpoint runs."""

    hosts_by_accession: dict[str, list[TokenStreamSlice]] = {}
    donors_by_accession: dict[str, list[TokenStreamSlice]] = {}
    for stream in host_streams:
        hosts_by_accession.setdefault(stream.accession, []).append(stream)
    for stream in donor_streams:
        donors_by_accession.setdefault(stream.accession, []).append(stream)
    pair_lookup = {pair.host_accession: pair for pair in donor_pairs}
    cases: list[FrozenAnomalyCase] = []
    sequences: dict[str, tuple[int, ...]] = {}
    calibration: dict[str, tuple[int, ...]] = {}
    for host_accession in sorted(pair_lookup):
        eligible = [
            (stream, calibration_start)
            for stream in hosts_by_accession.get(host_accession, ())
            if (calibration_start := canonical_host_calibration_start(stream, config)) is not None
        ]
        if not eligible:
            raise ValueError(
                f"host {host_accession} lacks a canonical evaluation prefix and "
                f"{config.native_calibration_segments}-segment calibration window"
            )
        host, calibration_start = sorted(
            eligible, key=lambda value: (-value[0].complete_segments, value[0].stream_id, value[1])
        )[0]
        calibration[host_accession] = tuple(host.token_ids[
            calibration_start * SEGMENT_TOKENS:
            (calibration_start + config.native_calibration_segments) * SEGMENT_TOKENS + 1
        ])
        pair = pair_lookup[host_accession]
        for distance, donor_accession, donor_ani in (
            ("near", pair.near_accession, pair.near_ani),
            ("far", pair.far_accession, pair.far_ani),
        ):
            donor_candidates = sorted(
                donors_by_accession.get(donor_accession, ()),
                key=lambda value: (-value.complete_segments, value.stream_id),
            )
            if not donor_candidates:
                raise ValueError(f"selected E25 donor {donor_accession} has no token stream")
            for depth, length in product(config.depths, config.lengths):
                host_dna = host.base_block(depth, length)
                if not _canonical_dna(host_dna):
                    raise ValueError(f"host replacement at {host_accession}/{depth}/{length} is noncanonical")
                host_gc = gc_fraction(host_dna)
                donor_choice = next(
                    ((stream, match) for stream in donor_candidates
                     if (match := _best_gc_block(stream, length, host_gc, tolerance=config.gc_tolerance))),
                    None,
                )
                if donor_choice is None:
                    raise ValueError(f"no GC-matched canonical donor block for {host_accession}/{distance}/{depth}/{length}")
                donor, (donor_start, donor_gc) = donor_choice
                same = _best_gc_block(
                    host, length, donor_gc, tolerance=config.gc_tolerance,
                    excluded=((depth, depth + length), (
                        calibration_start, calibration_start + config.native_calibration_segments
                    )),
                )
                if same is None:
                    raise ValueError(f"no GC-matched same-host relocation for {host_accession}/{depth}/{length}")
                same_start, same_gc = same
                end = depth + length + config.recovery_segments
                untouched = tuple(host.token_ids[: end * SEGMENT_TOKENS + 1])
                final_token = end * SEGMENT_TOKENS + 1
                untouched_dna = _token_dna(host, 0, final_token)
                prefix_dna = _token_dna(host, 0, depth * SEGMENT_TOKENS)
                replaced_dna = host.base_block(depth, length)
                donor_dna = donor.base_block(donor_start, length)
                relocated_dna = host.base_block(same_start, length)
                if len({len(replaced_dna), len(donor_dna), len(relocated_dna)}) != 1:
                    raise ValueError("replacement fragments must preserve exact base length")
                suffix_dna = _token_dna(
                    host, (depth + length) * SEGMENT_TOKENS, final_token
                )
                foreign_dna = prefix_dna + donor_dna + suffix_dna
                same_host_dna = prefix_dna + relocated_dna + suffix_dna
                foreign = list(untouched)
                left, width = depth * SEGMENT_TOKENS, length * SEGMENT_TOKENS
                foreign[left:left + width] = donor.token_block(donor_start, length)
                relocated = list(untouched)
                relocated[left:left + width] = host.token_block(same_start, length)
                arrays = {f"foreign_{distance}": tuple(foreign), "same_host": tuple(relocated), "untouched": untouched}
                identity = f"{host_accession}|{distance}|{depth}|{length}|{config.seed}"
                case = FrozenAnomalyCase(
                    case_id="anomaly_" + hashlib.sha256(identity.encode()).hexdigest()[:20],
                    host_stream_id=host.stream_id, host_accession=host_accession,
                    donor_stream_id=donor.stream_id, donor_accession=donor_accession,
                    donor_distance=distance, donor_ani=donor_ani, insertion_depth=depth,
                    length_segments=length, donor_start_segment=donor_start,
                    same_host_start_segment=same_start,
                    native_calibration_start_segment=calibration_start, donor_gc=donor_gc,
                    replaced_host_gc=host_gc, same_host_gc=same_gc,
                    token_sha256={label: _token_hash(value) for label, value in arrays.items()},
                    dna_sha256={
                        f"foreign_{distance}": hashlib.sha256(foreign_dna.encode("ascii")).hexdigest(),
                        "same_host": hashlib.sha256(same_host_dna.encode("ascii")).hexdigest(),
                        "untouched": hashlib.sha256(untouched_dna.encode("ascii")).hexdigest(),
                    },
                    source_coordinate_sha256={
                        "host_replaced": _coordinate_hash(host, depth, length),
                        "donor_fragment": _coordinate_hash(donor, donor_start, length),
                        "same_host_fragment": _coordinate_hash(host, same_start, length),
                        "native_calibration": _coordinate_hash(host, calibration_start, config.native_calibration_segments),
                    },
                )
                cases.append(case)
                sequences.update({f"{case.case_id}|{label}": value for label, value in arrays.items()})
    if len({case.host_accession for case in cases}) != config.hosts:
        raise ValueError(f"frozen panel requires exactly {config.hosts} eligible hosts")
    return tuple(cases), sequences, calibration


def validate_checkpoint_paths(paths: Mapping[str, str | Path]) -> dict[str, str]:
    if set(paths) != set(CHECKPOINT_SHA256):
        raise ValueError("both and only C16 and C19 checkpoints are required")
    actual = {model: sha256_file(path) for model, path in paths.items()}
    for model, expected in CHECKPOINT_SHA256.items():
        if actual[model] != expected:
            raise ValueError(f"{model} checkpoint SHA-256 mismatch")
    return actual


def verify_byte_identical_cases(frames: Mapping[str, pd.DataFrame]) -> None:
    """Refuse model comparison unless every materialized case hash is identical."""

    if set(frames) != {"C16", "C19"}:
        raise ValueError("paired comparison requires C16 and C19")
    columns = ["case_id", "sequence_class", "sequence_sha256"]
    identities = []
    for model in ("C16", "C19"):
        missing = set(columns) - set(frames[model])
        if missing:
            raise ValueError(f"{model} case table is missing {sorted(missing)}")
        identities.append(
            frames[model][columns].drop_duplicates().sort_values(columns[:2]).reset_index(drop=True)
        )
    if not identities[0].equals(identities[1]):
        raise ValueError("C16/C19 case hashes differ; protocol-mismatched comparison refused")


def robust_z(values: Sequence[float], native_values: Sequence[float], *, epsilon: float = 1e-9) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    native = np.asarray(native_values, dtype=float)
    if native.size == 0 or not np.isfinite(native).all() or not np.isfinite(values_array).all():
        raise ValueError("robust normalization requires finite values and native calibration data")
    median = np.median(native)
    mad = np.median(np.abs(native - median))
    return (values_array - median) / (1.4826 * mad + epsilon)


def kmer_distribution(sequence: str, k: int) -> dict[str, float]:
    sequence = sequence.upper()
    if k < 1 or not _canonical_dna(sequence) or len(sequence) < k:
        return {}
    counts: dict[str, int] = {}
    for index in range(len(sequence) - k + 1):
        word = sequence[index:index + k]
        counts[word] = counts.get(word, 0) + 1
    total = sum(counts.values())
    return {word: count / total for word, count in counts.items()}


def jensen_shannon(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    p = np.asarray([left.get(key, 0.0) for key in keys], dtype=float)
    q = np.asarray([right.get(key, 0.0) for key in keys], dtype=float)
    midpoint = (p + q) / 2
    divergence = 0.0
    for values in (p, q):
        active = values > 0
        divergence += 0.5 * float(np.sum(values[active] * np.log2(values[active] / midpoint[active])))
    return divergence


def markov_nll(sequence: str, background: str, *, maximum_order: int = 5, pseudocount: float = 0.5) -> float:
    """Interpolated order-0...5 host Markov negative log likelihood per base."""

    sequence, background = sequence.upper(), background.upper()
    if not _canonical_dna(sequence) or not _canonical_dna(background):
        raise ValueError("Markov scoring requires canonical DNA")
    total_nll = 0.0
    for index, base in enumerate(sequence):
        probabilities: list[float] = []
        for order in range(min(maximum_order, index) + 1):
            context = sequence[index - order:index]
            denominator = pseudocount * 4
            numerator = pseudocount
            for location in range(order, len(background)):
                if background[location - order:location] == context:
                    denominator += 1
                    numerator += int(background[location] == base)
            probabilities.append(numerator / denominator)
        total_nll -= math.log(sum(probabilities) / len(probabilities))
    return total_nll / max(len(sequence), 1)


def classical_sequence_features(sequence: str, preceding_host: str) -> dict[str, float]:
    if not _canonical_dna(sequence) or not _canonical_dna(preceding_host):
        raise ValueError("classical features require canonical DNA")
    result = {"gc_deviation": gc_fraction(sequence) - gc_fraction(preceding_host)}
    for k in range(1, 7):
        result[f"jsd_k{k}"] = jensen_shannon(
            kmer_distribution(sequence, k), kmer_distribution(preceding_host, k)
        )
    result["tetranucleotide_distance"] = math.sqrt(result["jsd_k4"])
    result["markov_nll"] = markov_nll(sequence, preceding_host)
    return result


def add_state_changes(frame: pd.DataFrame, *, group_columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in ("surprise_norm", "state_drift_norm"):
        if column in result:
            result[f"{column}_diff"] = result.groupby(list(group_columns), sort=False)[column].diff()
            result[f"{column}_log1p_diff"] = result.groupby(list(group_columns), sort=False)[column].transform(
                lambda values: np.log1p(values).diff()
            )
    return result


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Holm adjustment requires finite p-values")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def exact_sign_permutation_p(differences: Sequence[float]) -> float:
    values = np.asarray([value for value in differences if value != 0], dtype=float)
    if values.size == 0:
        return 1.0
    observed = abs(float(values.mean()))
    if values.size <= 20:
        statistics = [abs(float(np.mean(values * signs))) for signs in product((-1, 1), repeat=len(values))]
        return sum(value >= observed - 1e-15 for value in statistics) / len(statistics)
    # Exact enumeration is infeasible beyond 20 hosts; the preregistered full
    # study has eight, so this guard prevents accidental pseudoreplication.
    raise ValueError("exact paired sign permutation is limited to 20 host aggregates")


def rank_biserial(differences: Sequence[float]) -> float:
    values = np.asarray([value for value in differences if value != 0], dtype=float)
    if values.size == 0:
        return 0.0
    order = np.argsort(np.abs(values), kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    positive, negative = ranks[values > 0].sum(), ranks[values < 0].sum()
    return float((positive - negative) / ranks.sum())


def paired_effect(
    values: Mapping[str, float], *, seed: int = 20260813, samples: int = 2000
) -> dict[str, float | int]:
    hosts = sorted(values)
    observed = np.asarray([values[host] for host in hosts], dtype=float)
    if not hosts or not np.isfinite(observed).all():
        raise ValueError("paired effect requires finite host-level differences")
    rng = np.random.default_rng(seed)
    draws = np.median(observed[rng.integers(0, len(hosts), (samples, len(hosts)))], axis=1)
    scale = observed.std(ddof=1) if len(observed) > 1 else 0.0
    return {
        "sample_unit": "host", "hosts": len(hosts), "median_difference": float(np.median(observed)),
        "standardized_paired_effect": float(observed.mean() / scale) if scale else 0.0,
        "rank_biserial": rank_biserial(observed),
        "lower_95": float(np.quantile(draws, 0.025)), "upper_95": float(np.quantile(draws, 0.975)),
        "p_value": exact_sign_permutation_p(observed),
    }


def peak_enrichment(
    scores: Sequence[float], mask: Sequence[bool], *, permutations: int = 10_000, seed: int = 20260813
) -> dict[str, object]:
    values, anomaly = np.asarray(scores, dtype=float), np.asarray(mask, dtype=bool)
    if values.size != anomaly.size or not anomaly.any() or anomaly.all() or not np.isfinite(values).all():
        raise ValueError("peak enrichment requires aligned finite scores and a partial anomaly mask")
    peak = int(np.argmax(values))
    native = values[~anomaly]
    peak_z = float(robust_z([values[peak]], native)[0])

    def statistic(candidate: np.ndarray) -> tuple[int, dict[str, float]]:
        overlaps: dict[str, float] = {}
        for proportion in (0.01, 0.05, 0.10):
            count = max(1, int(math.ceil(len(values) * proportion)))
            selected = np.argpartition(values, -count)[-count:]
            overlaps[f"top_{int(proportion * 100)}pct_overlap"] = float(candidate[selected].mean())
        return int(candidate[peak]), overlaps

    observed_capture, observed_overlap = statistic(anomaly)
    rng = random.Random(seed)
    possible = list(range(1, len(values)))
    shifts = possible if permutations >= len(possible) else [rng.choice(possible) for _ in range(permutations)]
    null = [statistic(np.roll(anomaly, shift)) for shift in shifts]
    result: dict[str, object] = {
        "global_peak_index": peak, "global_peak_height": float(values[peak]),
        "peak_to_native_robust_z": peak_z, "maximum_inside_insert": bool(observed_capture),
        **observed_overlap,
        "circular_shifts": len(shifts),
        "peak_capture_p_value": (1 + sum(item[0] >= observed_capture for item in null)) / (len(null) + 1),
    }
    for key, observed in observed_overlap.items():
        result[f"{key}_p_value"] = (1 + sum(item[1][key] >= observed for item in null)) / (len(null) + 1)
    return result


def lagged_spearman(left: Sequence[float], right: Sequence[float], lags: Iterable[int] = range(-4, 9)) -> dict[int, float]:
    from scipy.stats import rankdata

    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if len(x) != len(y):
        raise ValueError("lagged arrays must align")
    result: dict[int, float] = {}
    for lag in lags:
        a, b = (x[-lag:], y[:lag]) if lag < 0 else ((x[:-lag], y[lag:]) if lag > 0 else (x, y))
        result[int(lag)] = float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]) if len(a) >= 3 else math.nan
    return result


def nested_leave_one_host_out(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    host_column: str = "host_accession",
    label_column: str = "is_foreign",
    native_column: str = "is_native_calibration",
    regularization_grid: Sequence[float] = REGULARIZATION_GRID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """L2 logistic regression with normalization and tuning confined to training hosts."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    required = {host_column, label_column, native_column, *features}
    if missing := required - set(frame):
        raise ValueError(f"detector frame is missing {sorted(missing)}")
    if not np.isfinite(frame[list(features)].to_numpy(dtype=float)).all():
        raise ValueError("detector features must be finite")
    predictions: list[pd.DataFrame] = []
    coefficients: list[dict[str, object]] = []
    hosts = sorted(frame[host_column].astype(str).unique())
    if len(hosts) < 3:
        raise ValueError("nested leave-one-host-out requires at least three hosts")
    for held_out in hosts:
        train = frame[frame[host_column].astype(str).ne(held_out)].copy()
        test = frame[frame[host_column].astype(str).eq(held_out)].copy()
        if int(train[native_column].sum()) < 100:
            raise ValueError("every outer fold requires at least 100 unique native calibration negatives")
        native = train[train[native_column].astype(bool)]
        medians = native[list(features)].median()
        mads = (native[list(features)] - medians).abs().median()
        scales = 1.4826 * mads + 1e-9
        train_x = ((train[list(features)] - medians) / scales).to_numpy()
        test_x = ((test[list(features)] - medians) / scales).to_numpy()
        train_y = train[label_column].astype(int).to_numpy()
        inner_hosts = sorted(train[host_column].astype(str).unique())
        grid_scores: dict[float, list[float]] = {float(value): [] for value in regularization_grid}
        for validation_host in inner_hosts:
            fit_mask = train[host_column].astype(str).ne(validation_host).to_numpy()
            validation_mask = ~fit_mask
            if len(np.unique(train_y[fit_mask])) < 2 or len(np.unique(train_y[validation_mask])) < 2:
                continue
            for c_value in regularization_grid:
                estimator = LogisticRegression(C=float(c_value), solver="liblinear", random_state=0)
                estimator.fit(train_x[fit_mask], train_y[fit_mask])
                grid_scores[float(c_value)].append(average_precision_score(
                    train_y[validation_mask], estimator.predict_proba(train_x[validation_mask])[:, 1]
                ))
        chosen = max(grid_scores, key=lambda value: (np.mean(grid_scores[value]) if grid_scores[value] else -math.inf, -value))
        estimator = LogisticRegression(C=chosen, solver="liblinear", random_state=0)
        estimator.fit(train_x, train_y)
        fold = test.copy()
        fold["outer_host"] = held_out
        fold["prediction"] = estimator.predict_proba(test_x)[:, 1]
        training_scores = estimator.predict_proba(train_x)[:, 1]
        training_native = train[native_column].astype(bool).to_numpy()
        fold["training_native_threshold_1pct_fpr"] = float(np.nextafter(
            np.quantile(training_scores[training_native], 0.99, method="higher"), np.inf
        ))
        fold["regularization_c"] = chosen
        predictions.append(fold)
        coefficients.extend(
            {"outer_host": held_out, "feature": feature, "coefficient": float(value), "regularization_c": chosen}
            for feature, value in zip(features, estimator.coef_[0])
        )
        coefficients.append({"outer_host": held_out, "feature": "intercept", "coefficient": float(estimator.intercept_[0]), "regularization_c": chosen})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(coefficients)


def validate_study_grid(
    cases: pd.DataFrame,
    segments: pd.DataFrame,
    blocks: pd.DataFrame,
    config: ScientificStudyConfig,
) -> None:
    """Fail closed on incomplete, non-finite, or protocol-mismatched bundles."""

    expected = set(product(ANOMALY_CLASSES[:2], config.lengths, config.depths))
    hosts = sorted(cases["host_accession"].astype(str).unique())
    if len(hosts) != config.hosts:
        raise ValueError(f"{config.mode} comparison requires {config.hosts} complete hosts")
    for host in hosts:
        observed = set(map(tuple, cases.loc[
            cases["host_accession"].astype(str).eq(host), ["sequence_class", "length_segments", "insertion_depth"]
        ].drop_duplicates().itertuples(index=False, name=None)))
        if observed != expected:
            raise ValueError(f"host {host} does not contain the exact anomaly grid")
    for name, table in (("segments", segments), ("blocks", blocks)):
        numeric = table.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} contains NaN or infinite telemetry")
    if "normalized_depth" not in blocks or not blocks["normalized_depth"].between(0, 1).all():
        raise ValueError("block table lacks valid normalized network depth")


def validate_full_grid(cases: pd.DataFrame, segments: pd.DataFrame, blocks: pd.DataFrame) -> None:
    """Backward-compatible exact full-grid validator."""

    validate_study_grid(cases, segments, blocks, ScientificStudyConfig.full())


def analysis_contract() -> dict[str, object]:
    return {
        "study_version": STUDY_VERSION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "bounded_config": ScientificStudyConfig.bounded().to_dict(),
        "bounded_workload": planned_segment_forwards(ScientificStudyConfig.bounded()),
        "full_config": ScientificStudyConfig.full().to_dict(),
        "anomaly_classes": ANOMALY_CLASSES,
        "memory_interventions": MEMORY_INTERVENTIONS,
        "needle_controls": NEEDLE_CONTROLS,
        "detectors": {
            "reference": REFERENCE_FEATURES,
            "memory": MEMORY_FEATURES,
            "counterfactual": COUNTERFACTUAL_FEATURES,
            "regularization_grid": REGULARIZATION_GRID,
            "validation": "nested_leave_one_host_out",
        },
        "sample_unit": "host",
        "test_panel": "excluded",
        "model_contrast_label": "combined checkpoint/exposure progression",
    }
