"""Build and evaluate the frozen C16/C19 anomaly-and-needle validation panel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from seqtrainer.data.bacteria_titan import (
    StageCPanelManifest, TokenStreamDataset, validate_panel_against_dataset,
)

from .anomaly_study import (
    CHECKPOINT_SHA256, COUNTERFACTUAL_FEATURES, MEMORY_FEATURES,
    REFERENCE_FEATURES, STUDY_VERSION,
    FrozenAnomalyCase, ScientificStudyConfig, analysis_contract,
    add_state_changes, canonical_host_calibration_start, choose_relative_donors, contract_hash,
    freeze_anomaly_panel,
    classical_sequence_features, evaluation_interventions, holm_adjust, sha256_file,
    lagged_spearman, nested_leave_one_host_out, paired_effect, peak_enrichment, validate_study_grid,
    planned_segment_forwards, runtime_projection, verify_byte_identical_cases,
)
from .checkpoints import checkpoint_parent_dataset_fingerprint
from .config import MemoryMode, StageCModelConfig
from .context_eval import (
    CaseResumeStore, ContextEvalConfig, NeedleCase, contract_hash as legacy_contract_hash,
    needle_case_bundle_for_host,
    write_jsonl, write_sequence_slices,
)
from .context_eval_cli import _load_checkpoint, _run_needle_case, _score_segments, _slices, _tokenizer, _warm
from .model import StageCPaperMACForCausalLM, detach_stream_states


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(partial, path)


def _read_membership(path: Path) -> dict[str, str]:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, sep=None, engine="python")
    if not {"accession", "ani_cluster_99"}.issubset(frame):
        raise ValueError("ANI membership requires accession and ani_cluster_99")
    if frame.groupby("accession")["ani_cluster_99"].nunique().gt(1).any():
        raise ValueError("ANI99 membership is not unique")
    return dict(zip(frame["accession"].astype(str), frame["ani_cluster_99"].astype(str)))


def _read_ani(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix in {".tsv", ".txt"} else ",")


def _write_npz(path: Path, arrays: Mapping[str, Sequence[int]]) -> None:
    # np.savez writes directly, so create it in the same directory and replace.
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(partial, **{key: np.asarray(value, dtype=np.int64) for key, value in sorted(arrays.items())})
    os.replace(partial, path)


def freeze(args: argparse.Namespace) -> Path:
    config = (
        ScientificStudyConfig.full() if args.mode == "full"
        else ScientificStudyConfig.bounded() if args.mode == "bounded"
        else ScientificStudyConfig()
    )
    validation = StageCPanelManifest.from_path(args.validation_panel)
    training = StageCPanelManifest.from_path(args.e25_panel)
    if validation.payload["role"] != "validation" or training.payload["role"] != "train":
        raise ValueError("03q requires a validation host panel and the exact E25 training donor panel")
    final_output = args.output
    input_contract = {
        "study_version": STUDY_VERSION,
        "mode": config.mode,
        "dataset_manifest_sha256": sha256_file(args.dataset_dir / "token_stream_manifest.json"),
        "validation_panel_sha256": validation.hash,
        "e25_training_panel_sha256": training.hash,
        "ani_pairs_sha256": sha256_file(args.ani_pairs),
        "ani_membership_sha256": sha256_file(args.ani_membership),
    }
    existing_manifest = final_output / "frozen_panel_manifest.json"
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text())
        if any(existing.get(key) != value for key, value in input_contract.items()):
            raise ValueError("existing frozen panel has a different immutable input contract")
        artifacts = existing.get("artifact_sha256", {})
        if not artifacts or any(
            not (final_output / name).is_file()
            or sha256_file(final_output / name) != digest
            for name, digest in artifacts.items()
        ):
            raise ValueError("existing frozen panel artifact changed or is missing")
        return final_output

    dataset = TokenStreamDataset(args.dataset_dir, verify_checksums=True)
    validate_panel_against_dataset(validation, dataset)
    validate_panel_against_dataset(training, dataset)
    tokenizer_name = json.loads((args.dataset_dir / "token_stream_manifest.json").read_text())["tokenizer"]["name"]
    tokenizer = _tokenizer(tokenizer_name)
    hosts = _slices(dataset, validation, "val", tokenizer.decode)
    donors = _slices(dataset, training, "train", tokenizer.decode)
    groups = _read_membership(args.ani_membership)
    needle_config = ContextEvalConfig(
        hosts=config.hosts, insertion_segments=(1,), needle_distances=config.needle_distances,
        distractor_counts=config.needle_distractors, warmup_segments=16,
        recovery_segments=16, gc_tolerance=config.gc_tolerance,
    )
    eligible_host_accessions = {
        value.accession for value in hosts.values()
        if canonical_host_calibration_start(value, config) is not None
    }
    ordered_hosts = sorted(
        eligible_host_accessions,
        key=lambda value: contract_hash({"seed": config.seed, "accession": value}),
    )
    donor_accessions = sorted({value.accession for value in donors.values()})
    ani = _read_ani(args.ani_pairs)
    selected_pairs, selection_rejections, needle_bundles = [], [], {}
    for host in ordered_hosts:
        try:
            pair = choose_relative_donors([host], donor_accessions, ani, groups)[0]
            host_stream = sorted(
                (
                    value for value in hosts.values()
                    if value.accession == host
                    and canonical_host_calibration_start(value, config) is not None
                ),
                key=lambda value: (
                    -value.complete_segments, value.stream_id,
                    canonical_host_calibration_start(value, config),
                ),
            )[0]
            needle_bundles[host] = needle_case_bundle_for_host(
                host_stream, tuple(hosts.values()), needle_config
            )
        except (IndexError, ValueError) as error:
            selection_rejections.append({"host_accession": host, "reason": str(error)})
            continue
        selected_pairs.append(pair)
        if len(selected_pairs) == config.hosts:
            break
    if len(selected_pairs) != config.hosts:
        raise ValueError(
            f"only {len(selected_pairs)} validation hosts satisfy the joint canonical contract; "
            f"rejections={selection_rejections[:8]}"
        )
    selected_accessions = {pair.host_accession for pair in selected_pairs}
    selected_streams = [value for value in hosts.values() if value.accession in selected_accessions]
    cases, arrays, calibration = freeze_anomaly_panel(
        selected_streams, tuple(donors.values()), selected_pairs, config
    )
    frozen_host_stream_ids = {case.host_stream_id for case in cases}
    needle_streams = sorted(
        (value for value in selected_streams if value.stream_id in frozen_host_stream_ids),
        key=lambda value: contract_hash({"seed": needle_config.seed + 1, "stream_id": value.stream_id}),
    )
    needle_cases = tuple(case for host in needle_streams for case in needle_bundles[host.accession][0])
    host_lookup = {value.stream_id: value for value in hosts.values()}
    needle_arrays = {
        f"{case_id}|needle": sequence
        for host in needle_streams
        for case_id, sequence in needle_bundles[host.accession][1].items()
    }
    wrong_arrays = {}
    wrong_segments = max(max(config.depths) - config.pre_segments, 0)
    for case in cases:
        wrong = host_lookup[case.wrong_host_stream_id]
        wrong_arrays[f"{case.case_id}|wrong_host_warmup"] = tuple(
            wrong.token_ids[:wrong_segments * 32]
        )
    for case in needle_cases:
        wrong = host_lookup[case.wrong_host_stream_id]
        wrong_arrays[f"{case.case_id}|wrong_host_warmup"] = tuple(
            wrong.token_ids[:case.query_segment * 32]
        )
    frozen_arrays = {
        **arrays, **needle_arrays, **wrong_arrays,
        **{f"native|{key}": value for key, value in calibration.items()},
    }
    output = final_output.with_name(final_output.name + ".partial")
    if output.exists():
        shutil.rmtree(output)
    if final_output.exists():
        shutil.rmtree(final_output)
    output.mkdir(parents=True)
    write_jsonl(output / "anomaly_cases.jsonl", (case.to_dict() for case in cases))
    write_jsonl(output / "needle_cases.jsonl", (case.to_dict() for case in needle_cases))
    pd.DataFrame([case.to_dict() for case in cases]).to_parquet(output / "anomaly_cases.parquet", index=False)
    pd.DataFrame([case.to_dict() for case in needle_cases]).to_parquet(output / "needle_cases.parquet", index=False)
    _write_npz(output / "token_arrays.npz", frozen_arrays)
    retained = [(key, tokenizer.decode(value)) for key, value in frozen_arrays.items()]
    noncanonical = {
        key: sorted(set(dna.upper()) - set("ACGT"))
        for key, dna in retained if set(dna.upper()) - set("ACGT")
    }
    if noncanonical:
        preview = dict(list(sorted(noncanonical.items()))[:8])
        raise ValueError(f"frozen retained FASTA contains noncanonical DNA: {preview}")
    retained_lookup = dict(retained)
    for case in cases:
        for label, expected in case.dna_sha256.items():
            actual = hashlib.sha256(retained_lookup[f"{case.case_id}|{label}"].encode("ascii")).hexdigest()
            if actual != expected:
                raise ValueError(f"retained DNA identity mismatch for {case.case_id}/{label}")
        actual_wrong = hashlib.sha256(
            retained_lookup[f"{case.case_id}|wrong_host_warmup"].encode("ascii")
        ).hexdigest()
        if actual_wrong != case.wrong_host_warmup_sha256:
            raise ValueError(f"retained wrong-host identity mismatch for {case.case_id}")
    for case in needle_cases:
        for label, expected in case.retained_slice_sha256.items():
            retained_label = "needle" if label == "needle_window" else label
            actual = hashlib.sha256(
                retained_lookup[f"{case.case_id}|{retained_label}"].encode("ascii")
            ).hexdigest()
            if actual != expected:
                raise ValueError(f"retained needle identity mismatch for {case.case_id}/{label}")
    write_sequence_slices(output / "retained_sequences.fasta.gz", retained)
    artifact_names = (
        "anomaly_cases.jsonl", "needle_cases.jsonl", "anomaly_cases.parquet",
        "needle_cases.parquet", "token_arrays.npz", "retained_sequences.fasta.gz",
    )
    manifest = {
        **input_contract, "format_version": 2,
        "study_version": STUDY_VERSION, "mode": config.mode,
        "config": config.to_dict(), "analysis_contract": analysis_contract(),
        "planned_workload": planned_segment_forwards(config),
        "hosts": sorted(selected_accessions), "donor_pairs": [pair.__dict__ for pair in selected_pairs],
        "canonical_selection": {
            "policy_version": 2, "alphabet": "ACGT",
            "candidate_order": "seeded_contract_hash_then_stable_token_order",
            "distractor_policy": "same_host_canonical_equal_base_length_then_next_host",
            "wrong_host_policy": "canonical_prefix_seeded_order",
            "evaluation_prefix_segments": (
                max(config.depths) + max(config.lengths) + config.recovery_segments
            ),
            "native_calibration_segments": config.native_calibration_segments,
            "calibration_policy": "earliest_nonoverlapping_canonical_window",
            "rejected_hosts": selection_rejections,
        },
        "case_list_sha256": contract_hash([case.to_dict() for case in cases]),
        "needle_list_sha256": contract_hash([case.to_dict() for case in needle_cases]),
        "token_arrays_sha256": sha256_file(output / "token_arrays.npz"),
        "artifact_sha256": {name: sha256_file(output / name) for name in artifact_names},
        "test_panel": "not accepted by this workflow",
    }
    manifest["panel_contract_sha256"] = contract_hash(manifest)
    _atomic_json(output / "frozen_panel_manifest.json", manifest)
    os.replace(output, final_output)
    return final_output


def estimate_runtime(args: argparse.Namespace) -> Path:
    """Measure the installed C19 evaluator and fail before an over-budget run."""

    panel_manifest = json.loads((args.frozen_panel / "frozen_panel_manifest.json").read_text())
    if panel_manifest["mode"] != "bounded":
        raise ValueError("the sub-24-hour runtime gate accepts only the bounded panel")
    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256["C19"]:
        raise ValueError("C19 immutable checkpoint SHA-256 mismatch")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type != "cuda":
        raise ValueError("runtime qualification requires CUDA")
    payload = _load_checkpoint(args.checkpoint, device, trusted=args.trust_owned_checkpoint)
    config = StageCModelConfig.from_dict(payload["model_config"])
    model = StageCPaperMACForCausalLM(config).to(device).eval()
    model.load_state_dict(payload["model_state"])
    arrays = np.load(args.frozen_panel / "token_arrays.npz")
    anomaly_key = next(key for key in sorted(arrays.files) if "|foreign_" in key)
    tokens = arrays[anomaly_key].tolist()
    probe_forwards = int(args.probe_forwards)
    adaptive_forwards = probe_forwards * 3 // 4
    no_memory_forwards = probe_forwards - adaptive_forwards

    def probe(count: int, mode: MemoryMode, stream_id: str) -> None:
        state = model.initial_states(stream_id)
        for index in range(count):
            start = (index % ((len(tokens) - 1) // 32)) * 32
            inputs = torch.tensor(tokens[start:start + 32], device=device).long()
            output = model.forward_segment((state,), inputs, memory_mode=mode)
            state = detach_stream_states(output.states[0])

    probe(4, MemoryMode.ADAPTIVE, "runtime:warmup")
    torch.cuda.synchronize(device)
    started = time.monotonic()
    probe(adaptive_forwards, MemoryMode.ADAPTIVE, "runtime:adaptive")
    probe(no_memory_forwards, MemoryMode.NONE, "runtime:none")
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    result = runtime_projection(
        planned_forwards=int(panel_manifest["planned_workload"]["total_segment_forwards"]),
        measured_segments_per_second=probe_forwards / elapsed,
        probe_forwards=probe_forwards,
        maximum_hours=float(args.max_hours),
    )
    result.update({
        "format_version": 1,
        "model": "C19",
        "checkpoint_sha256": CHECKPOINT_SHA256["C19"],
        "panel_contract_sha256": panel_manifest["panel_contract_sha256"],
        "device": torch.cuda.get_device_name(device),
        "measured_seconds": elapsed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    _atomic_json(args.output, result)
    if not result["accepted"]:
        raise RuntimeError(
            f"bounded C19 projection is {result['projected_hours']:.2f} h; "
            f"the limit is {result['maximum_hours']:.2f} h"
        )
    return args.output


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _case(value: Mapping[str, object]) -> FrozenAnomalyCase:
    return FrozenAnomalyCase(**value)  # type: ignore[arg-type]


def runtime_deadline_reached(
    started_monotonic: float,
    maximum_hours: float | None,
    *,
    now_monotonic: float | None = None,
) -> bool:
    if maximum_hours is None:
        return False
    now = time.monotonic() if now_monotonic is None else now_monotonic
    return now - started_monotonic >= maximum_hours * 3600.0


def run_model(args: argparse.Namespace) -> Path:
    panel_manifest = json.loads((args.frozen_panel / "frozen_panel_manifest.json").read_text())
    if sha256_file(args.frozen_panel / "token_arrays.npz") != panel_manifest["token_arrays_sha256"]:
        raise ValueError("frozen token arrays changed")
    if args.model not in CHECKPOINT_SHA256:
        raise ValueError("model must be C16 or C19")
    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256[args.model]:
        raise ValueError(f"{args.model} immutable checkpoint SHA-256 mismatch")
    output = args.output
    if (output / "COMPLETE.json").is_file():
        completed = json.loads((output / "model_run_manifest.json").read_text())
        expected = {
            "model": args.model,
            "checkpoint_sha256": CHECKPOINT_SHA256[args.model],
            "panel_contract_sha256": panel_manifest["panel_contract_sha256"],
            "case_list_sha256": panel_manifest["case_list_sha256"],
        }
        if any(completed.get(key) != value for key, value in expected.items()):
            raise ValueError("completed model run has a different immutable contract")
        for name, digest in completed.get("artifact_sha256", {}).items():
            if sha256_file(output / name) != digest:
                raise ValueError(f"completed model artifact changed: {name}")
        return output
    output.mkdir(parents=True, exist_ok=True)
    session_started = time.monotonic()
    paused_path = output / "PAUSED.json"
    if paused_path.exists():
        paused_path.unlink()

    def status(state: str, *, phase: str, case_id: str | None = None) -> None:
        _atomic_json(output / "LIVE_STATUS.json", {
            "format_version": 1, "state": state, "phase": phase,
            "case_id": case_id, "model": args.model,
            "elapsed_hours": (time.monotonic() - session_started) / 3600.0,
            "maximum_runtime_hours": args.max_runtime_hours,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    def should_pause(phase: str, case_id: str) -> bool:
        if not runtime_deadline_reached(session_started, args.max_runtime_hours):
            status("running", phase=phase, case_id=case_id)
            return False
        payload = {
            "format_version": 1, "state": "paused", "phase": phase,
            "case_id": case_id, "model": args.model,
            "elapsed_hours": (time.monotonic() - session_started) / 3600.0,
            "maximum_runtime_hours": args.max_runtime_hours,
            "resume": "rerun the identical run-model command",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(paused_path, payload)
        status("paused", phase=phase, case_id=case_id)
        return True

    status("running", phase="load")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    payload = _load_checkpoint(args.checkpoint, device, trusted=args.trust_owned_checkpoint)
    dataset_hash = sha256_file(args.dataset_dir / "token_stream_manifest.json")
    if checkpoint_parent_dataset_fingerprint(payload) != dataset_hash:
        raise ValueError("checkpoint and frozen panel dataset disagree")
    config = StageCModelConfig.from_dict(payload["model_config"])
    tokenizer = _tokenizer(config.tokenizer_name)
    model = StageCPaperMACForCausalLM(config).to(device).eval()
    model.load_state_dict(payload["model_state"])
    dataset = TokenStreamDataset(args.dataset_dir, verify_checksums=True)
    validation = StageCPanelManifest.from_path(args.validation_panel)
    slices = _slices(dataset, validation, "val", tokenizer.decode)
    arrays = np.load(args.frozen_panel / "token_arrays.npz")
    cases = [_case(value) for value in _read_jsonl(args.frozen_panel / "anomaly_cases.jsonl")]
    stream_lookup = {value.stream_id: value for value in slices.values()}
    store = CaseResumeStore(output / "resume", {
        "model": args.model, "checkpoint_sha256": CHECKPOINT_SHA256[args.model],
        "panel_contract_sha256": panel_manifest["panel_contract_sha256"],
    })
    all_rows: list[dict[str, object]] = []
    for case in cases:
        if should_pause("anomaly", case.case_id):
            return output
        if not store.completed(case.case_id):
            host = stream_lookup[case.host_stream_id]
            wrong = stream_lookup[case.wrong_host_stream_id]
            scoring_start = case.insertion_depth - int(panel_manifest["config"]["pre_segments"])
            carried = _warm(model, host.token_ids, scoring_start, stream_id=host.stream_id, device=device)
            wrong_tokens = arrays[f"{case.case_id}|wrong_host_warmup"].tolist()
            wrong_state = _warm(model, wrong_tokens, scoring_start, stream_id=wrong.stream_id, device=device)
            rows = []
            for sequence_class in (f"foreign_{case.donor_distance}", "same_host", "untouched"):
                tokens = arrays[f"{case.case_id}|{sequence_class}"].tolist()
                interventions = evaluation_interventions(panel_manifest["mode"], sequence_class)
                for intervention in interventions:
                    if intervention == "carried":
                        state, mode = carried, MemoryMode.ADAPTIVE
                    elif intervention == "reset":
                        state, mode = model.initial_states(case.case_id + ":reset"), MemoryMode.ADAPTIVE
                    elif intervention == "wrong_host":
                        state, mode = wrong_state, MemoryMode.ADAPTIVE
                    else:
                        state, mode = model.initial_states(case.case_id + ":none"), MemoryMode.NONE
                    scored = _score_segments(model, tokens, start_segment=scoring_start,
                                             initial_states=state, mode=mode, decode=tokenizer.decode, device=device)
                    for row in scored:
                        segment_index = int(row["segment_index"])
                        segment_dna = tokenizer.decode(tokens[
                            segment_index * 32:(segment_index + 1) * 32
                        ])
                        background = host.base_block(max(0, segment_index - 8), min(8, segment_index))
                        if not background:
                            background = host.base_block(0, 1)
                        rows.append({
                            "model": args.model, "benchmark": "anomaly", "case_id": case.case_id,
                            "host_accession": case.host_accession, "donor_accession": case.donor_accession,
                            "donor_distance": case.donor_distance, "donor_ani": case.donor_ani,
                            "sequence_class": sequence_class, "intervention": intervention,
                            "insertion_depth": case.insertion_depth, "length_segments": case.length_segments,
                            "relative_segment": int(row["segment_index"]) - case.insertion_depth,
                            "sequence_sha256": case.token_sha256[sequence_class],
                            **classical_sequence_features(segment_dna, background), **row,
                        })
            store.commit(case.case_id, {"rows": rows})
        all_rows.extend(store.load(case.case_id)["result"]["rows"])
    for host_accession in panel_manifest["hosts"]:
        native_id = f"native:{host_accession}"
        if should_pause("native_calibration", native_id):
            return output
        if not store.completed(native_id):
            tokens = arrays[f"native|{host_accession}"].tolist()
            state = model.initial_states(native_id)
            rows = _score_segments(model, tokens, start_segment=0, initial_states=state,
                                   mode=MemoryMode.ADAPTIVE, decode=tokenizer.decode, device=device)
            store.commit(native_id, {"rows": [{
                "model": args.model, "benchmark": "anomaly", "case_id": native_id,
                "host_accession": host_accession, "donor_accession": "",
                "donor_distance": "native", "donor_ani": 100.0,
                "sequence_class": "native_calibration", "intervention": "carried",
                "insertion_depth": -1, "length_segments": 0,
                "relative_segment": int(row["segment_index"]),
                "sequence_sha256": legacy_contract_hash(tokens),
                **classical_sequence_features(
                    tokenizer.decode(tokens[int(row["segment_index"]) * 32:(int(row["segment_index"]) + 1) * 32]),
                    tokenizer.decode(tokens[max(0, int(row["segment_index"]) - 8) * 32:
                                            max(1, int(row["segment_index"])) * 32]),
                ), **row,
            } for row in rows]})
        all_rows.extend(store.load(native_id)["result"]["rows"])
    needle_rows: list[dict[str, object]] = []
    for value in _read_jsonl(args.frozen_panel / "needle_cases.jsonl"):
        value["key_tokens"] = tuple(value["key_tokens"])
        value["value_tokens"] = tuple(value["value_tokens"])
        case = NeedleCase(**value)  # type: ignore[arg-type]
        if should_pause("needle", case.case_id):
            return output
        if not store.completed(case.case_id):
            store.commit(case.case_id, {"rows": _run_needle_case(
                case, stream_lookup, model, device,
                sequence_tokens=arrays[f"{case.case_id}|needle"].tolist(),
                wrong_host_tokens=arrays[f"{case.case_id}|wrong_host_warmup"].tolist(),
            )})
        for row in store.load(case.case_id)["result"]["rows"]:
            needle_rows.append({"model": args.model, **case.to_dict(), **row})
    block_rows, token_rows, segment_rows = [], [], []
    for row in all_rows:
        flat = dict(row)
        diagnostics = flat.pop("block_diagnostics")
        nll, bpbs = flat.pop("token_nll"), flat.pop("token_bpb")
        segment_rows.append(flat)
        identity = {key: flat[key] for key in (
            "model", "case_id", "host_accession", "sequence_class", "intervention",
            "relative_segment", "sequence_sha256", "donor_distance", "length_segments", "insertion_depth"
        )}
        block_rows.extend({**identity, **diagnostic} for diagnostic in diagnostics)
        token_rows.extend({**identity, "token_position": index, "token_nll_nats": left, "token_bpb": right}
                          for index, (left, right) in enumerate(zip(nll, bpbs)))
    tables = {
        "segment_metrics.parquet": pd.DataFrame(segment_rows),
        "token_metrics.parquet": pd.DataFrame(token_rows),
        "block_metrics.parquet": pd.DataFrame(block_rows),
        "needle_metrics.parquet": pd.DataFrame(needle_rows),
    }
    for name, table in tables.items():
        numeric = table.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"refuse completion: {name} contains NaN or infinite values")
        table.to_parquet(output / name, index=False)
    artifacts = {
        name: sha256_file(output / name) for name in (
            "segment_metrics.parquet", "token_metrics.parquet", "block_metrics.parquet",
            "needle_metrics.parquet",
        )
    }
    _atomic_json(output / "model_run_manifest.json", {
        "study_version": STUDY_VERSION, "model": args.model,
        "checkpoint_sha256": CHECKPOINT_SHA256[args.model],
        "panel_contract_sha256": panel_manifest["panel_contract_sha256"],
        "case_list_sha256": panel_manifest["case_list_sha256"], "artifact_sha256": artifacts,
        "complete_cases": len(cases),
        "elapsed_hours": (time.monotonic() - session_started) / 3600.0,
        "planned_workload": panel_manifest.get("planned_workload"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    _atomic_json(output / "COMPLETE.json", {"rows": len(segment_rows), "finite": True})
    status("completed", phase="complete")
    return output


def _detector_frame(frame: pd.DataFrame) -> pd.DataFrame:
    carried = frame[frame["intervention"].eq("carried")].copy()
    carried = add_state_changes(
        carried.sort_values(["model", "case_id", "sequence_class", "segment_index"]),
        group_columns=("model", "case_id", "sequence_class"),
    )
    key = ["model", "case_id", "sequence_class", "relative_segment"]
    reset = frame[frame["intervention"].eq("reset")][key + ["bpb", "surprise_norm"]].rename(
        columns={"bpb": "reset_bpb", "surprise_norm": "reset_surprise_norm"}
    )
    carried = carried.merge(reset, on=key, how="left")
    carried["reset_bpb"] = carried["reset_bpb"].fillna(carried["bpb"])
    carried["reset_surprise_norm"] = carried["reset_surprise_norm"].fillna(carried["surprise_norm"])
    native = carried["sequence_class"].eq("native_calibration")
    carried["is_native_calibration"] = native
    carried["is_foreign"] = (
        carried["sequence_class"].str.startswith("foreign_")
        & carried["relative_segment"].ge(0)
        & carried["relative_segment"].lt(carried["length_segments"])
    )
    carried["context_conflict"] = carried["bpb"] - carried["reset_bpb"]
    carried["carried_minus_reset_surprise_log1p"] = (
        np.log1p(carried["surprise_norm"]) - np.log1p(carried["reset_surprise_norm"])
    )
    carried["momentary_surprise_rms_max_log1p"] = np.log1p(carried["momentary_surprise_rms_max"])
    carried["surprise_norm_log1p_diff"] = carried["surprise_norm_log1p_diff"].fillna(0.0)
    carried["memory_update_norm_log1p"] = np.log1p(carried["memory_update_norm"])
    carried["retrieval_norm_log1p"] = np.log1p(carried["retrieval_norm"])
    carried["state_drift_norm_log1p_diff"] = carried["state_drift_norm_log1p_diff"].fillna(0.0)
    same_host = carried[carried["sequence_class"].eq("same_host")][
        ["model", "case_id", "relative_segment", "bpb"]
    ].rename(columns={"bpb": "same_host_bpb"})
    carried = carried.merge(same_host, on=["model", "case_id", "relative_segment"], how="left")
    carried["foreign_minus_same_host_bpb"] = (carried["bpb"] - carried["same_host_bpb"]).fillna(0.0)
    carried["same_host_bpb"] = carried["same_host_bpb"].fillna(carried["bpb"])
    return carried


def _performance(predictions: pd.DataFrame, detector: str, model: str) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    labels = predictions["is_foreign"].astype(int).to_numpy()
    scores = predictions["prediction"].to_numpy(dtype=float)
    native = predictions["is_native_calibration"].astype(bool).to_numpy()
    thresholds = predictions["training_native_threshold_1pct_fpr"].to_numpy(dtype=float)
    positives = labels.astype(bool)
    calibration = LogisticRegression(C=1e6, solver="liblinear").fit(
        np.log(np.clip(scores, 1e-9, 1 - 1e-9) / np.clip(1 - scores, 1e-9, 1))[..., None], labels
    )
    negative_bases = float(predictions.loc[native, "bases"].sum())
    return {
        "model": model, "detector": detector, "sample_unit": "host",
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
        "threshold_1pct_fpr": float(np.median(thresholds)),
        "tpr_at_1pct_fpr": float((scores[positives] >= thresholds[positives]).mean()),
        "false_positives_per_mb": float(
            (scores[native] >= thresholds[native]).sum() * 1_000_000 / negative_bases
        ),
        "brier_score": float(brier_score_loss(labels, scores)),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
    }


def _write_core_figures(
    output: Path,
    detector_frame: pd.DataFrame,
    performance: pd.DataFrame,
    blocks: pd.DataFrame,
    needles: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    figures = output / "figures"
    sources = output / "figure_sources"
    figures.mkdir(exist_ok=True)
    sources.mkdir(exist_ok=True)
    palette = {"C16": "#0072B2", "C19": "#D55E00"}
    definitions = {
        "01_panel_quality": ("donor_ani", "Donor ANI (%)"),
        "02_representative_tracks": ("bpb", "Bits per base"),
        "03_boundary_trajectories": ("momentary_surprise_rms_max", "Momentary surprise RMS"),
        "04_host_comparisons": ("memory_update_norm", "Memory update norm"),
        "05_correlation_dependence": ("surprise_norm", "Accumulated surprise norm"),
        "06_detection_performance": ("bpb", "Bits per base"),
        "07_distance_length_localization": ("context_conflict", "Context conflict BPB"),
        "08_surprise_peak_analysis": ("momentary_surprise_rms_max", "Momentary surprise RMS"),
    }
    for name, (column, ylabel) in definitions.items():
        source = detector_frame[["model", "host_accession", "relative_segment", "sequence_class", column]].copy()
        source.to_csv(sources / f"{name}.csv", index=False)
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        for model, values in source.groupby("model"):
            trajectory = values.groupby("relative_segment")[column].median().sort_index()
            axis.plot(trajectory.index, trajectory.values, label=model, color=palette.get(model))
        axis.axvspan(0, 1, color="#999999", alpha=0.16, label="insert onset")
        axis.set(xlabel="Relative segment", ylabel=ylabel, title=name.replace("_", " ").title())
        axis.legend(frameon=False)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(figures / f"{name}.{suffix}", dpi=180)
        plt.close(fig)
        (figures / f"{name}.caption.md").write_text(
            f"{ylabel} summarized by model. Shading marks the known synthetic onset; source: figure_sources/{name}.csv.\n"
        )
    for name, source, x, y, ylabel in (
        ("09_per_block_memory_response", blocks, "normalized_depth", "momentary_surprise_rms_max", "Momentary surprise RMS"),
        ("10_needle_performance", needles, "distance_segments", "carried_minus_reset_log_probability", "Carried−reset target log probability"),
    ):
        source.to_csv(sources / f"{name}.csv", index=False)
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        for model, values in source.groupby("model"):
            curve = values.groupby(x)[y].median().sort_index()
            axis.plot(curve.index, curve.values, marker="o", label=model, color=palette.get(model))
        axis.set(xlabel=x.replace("_", " ").title(), ylabel=ylabel, title=name.replace("_", " ").title())
        axis.legend(frameon=False)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(figures / f"{name}.{suffix}", dpi=180)
        plt.close(fig)
        (figures / f"{name}.caption.md").write_text(
            f"{ylabel} by model; source: figure_sources/{name}.csv. Blocks use normalized network depth.\n"
        )
    supplemental = {
        "s01_per_host_anomaly_tracks": "bpb",
        "s02_full_feature_violin_atlas": "momentary_surprise_rms_max",
        "s03_coefficient_stability": "theta_mean",
        "s04_native_false_positive_tracks": "context_conflict",
        "s05_jsd_k1_k6": "jsd_k6",
        "s06_numerical_gradient_diagnostics": "conditioned_gradient_rms_max",
        "s07_resume_audit": "finite",
    }
    for name, column in supplemental.items():
        source = detector_frame[["model", "host_accession", "relative_segment", "sequence_class", column]]
        source.to_csv(sources / f"{name}.csv", index=False)
        fig, axis = plt.subplots(figsize=(6.4, 3.6))
        for model, values in source.groupby("model"):
            curve = values.groupby("relative_segment")[column].mean().sort_index()
            axis.plot(curve.index, curve.values, color=palette.get(model), label=model)
        axis.set(xlabel="Relative segment", ylabel=column.replace("_", " "), title=name.replace("_", " ").title())
        axis.legend(frameon=False)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(figures / f"{name}.{suffix}", dpi=180)
        plt.close(fig)
        (figures / f"{name}.caption.md").write_text(
            f"Supplemental diagnostic; source: figure_sources/{name}.csv.\n"
        )
    performance.to_csv(sources / "detector_performance.csv", index=False)


def _write_dictionary(output: Path, frame: pd.DataFrame) -> None:
    rows = ["# Data dictionary", "", "| Column | dtype |", "|---|---|"]
    rows.extend(f"| `{column}` | `{dtype}` |" for column, dtype in frame.dtypes.items())
    (output / "DATA_DICTIONARY.md").write_text("\n".join(rows) + "\n")


def _correlation_payload(frame: pd.DataFrame, *, seed: int = 20260813) -> dict[str, object]:
    from scipy.stats import rankdata

    features = [name for name in (
        "bpb", "context_conflict", "momentary_surprise_rms_max", "surprise_norm",
        "memory_update_norm", "retrieval_norm", "state_drift_norm_diff", "theta_mean", "donor_ani",
    ) if name in frame]
    payload: dict[str, object] = {"interpretation": "mechanistic diagnostic; no causal claim", "groups": {}}
    rng = np.random.default_rng(seed)
    for model in ("C16", "C19"):
        for label, mask in (
            ("native", frame["sequence_class"].eq("native_calibration")),
            ("foreign_near", frame["sequence_class"].eq("foreign_near") & frame["is_foreign"]),
            ("foreign_far", frame["sequence_class"].eq("foreign_far") & frame["is_foreign"]),
        ):
            values = frame[frame["model"].eq(model) & mask]
            if len(values) < 3:
                continue
            ranks = pd.DataFrame({name: rankdata(values[name]) for name in features})
            matrix = ranks.corr(method="pearson")
            hosts = sorted(values["host_accession"].unique())
            target_draws: list[float] = []
            for _ in range(2000):
                sampled = rng.choice(hosts, len(hosts), replace=True)
                draw = pd.concat([values[values["host_accession"].eq(host)] for host in sampled])
                target_draws.append(float(draw["bpb"].corr(draw["momentary_surprise_rms_max"], method="spearman")))
            finite_draws = np.asarray(target_draws, dtype=float)
            finite_draws = finite_draws[np.isfinite(finite_draws)]
            controls = pd.DataFrame({
                "length": values["length_segments"].astype(float),
                "depth": values["insertion_depth"].astype(float),
                "position": values["relative_segment"].astype(float),
                "gc": values["gc_deviation"].astype(float),
            }, index=values.index)
            controls = pd.concat(
                [controls, pd.get_dummies(values["host_accession"], prefix="host", dtype=float)], axis=1
            ).fillna(0.0)
            design = np.column_stack([np.ones(len(controls)), controls.to_numpy(dtype=float)])

            def partial(left: str, right: str) -> float | None:
                left_rank, right_rank = rankdata(values[left]), rankdata(values[right])
                left_residual = left_rank - design @ np.linalg.lstsq(design, left_rank, rcond=None)[0]
                right_residual = right_rank - design @ np.linalg.lstsq(design, right_rank, rcond=None)[0]
                correlation = float(np.corrcoef(left_residual, right_residual)[0, 1])
                return correlation if math.isfinite(correlation) else None

            pairs = (
                ("bpb", "momentary_surprise_rms_max"), ("bpb", "surprise_norm"),
                ("context_conflict", "momentary_surprise_rms_max"),
                ("momentary_surprise_rms_max", "memory_update_norm"),
                ("retrieval_norm", "bpb"), ("state_drift_norm_diff", "bpb"),
                ("donor_ani", "context_conflict"),
            )
            payload["groups"][f"{model}/{label}"] = {
                "features": features,
                "spearman": [
                    [float(value) if math.isfinite(float(value)) else None for value in row]
                    for row in matrix.to_numpy()
                ],
                "bpb_momentary_surprise_host_bootstrap_95": [
                    float(np.quantile(finite_draws, 0.025)), float(np.quantile(finite_draws, 0.975))
                ] if len(finite_draws) else None,
                "partial_rank_correlations": {
                    f"{left}__{right}": partial(left, right) for left, right in pairs
                },
            }
    return payload


def available_analysis_models(root: Path) -> tuple[str, ...]:
    """Return the C19-first model set eligible for scientific analysis."""

    if not (root / "C19" / "COMPLETE.json").is_file():
        raise ValueError("refuse analysis: the primary C19 run is incomplete")
    return ("C19", "C16") if (root / "C16" / "COMPLETE.json").is_file() else ("C19",)


def compare(args: argparse.Namespace) -> Path:
    models = available_analysis_models(args.input)
    outputs = {model: args.input / model for model in models}
    for model, directory in outputs.items():
        if not (directory / "COMPLETE.json").is_file():
            raise ValueError(f"refuse comparison: {model} run is incomplete")
        manifest = json.loads((directory / "model_run_manifest.json").read_text())
        if manifest["checkpoint_sha256"] != CHECKPOINT_SHA256[model]:
            raise ValueError(f"refuse comparison: {model} checkpoint mismatch")
    manifests = [json.loads((outputs[model] / "model_run_manifest.json").read_text()) for model in models]
    if len(manifests) == 2 and manifests[0]["panel_contract_sha256"] != manifests[1]["panel_contract_sha256"]:
        raise ValueError("refuse comparison: panel contracts differ")
    frames = {model: pd.read_parquet(outputs[model] / "segment_metrics.parquet") for model in outputs}
    if len(models) == 2:
        verify_byte_identical_cases(frames)
    combined = pd.concat(frames.values(), ignore_index=True)
    panel_root = args.input / "frozen_panel"
    panel_manifest = json.loads((panel_root / "frozen_panel_manifest.json").read_text())
    if panel_manifest["mode"] in {"bounded", "full"}:
        case_frame = pd.DataFrame(_read_jsonl(panel_root / "anomaly_cases.jsonl"))
        case_frame["sequence_class"] = "foreign_" + case_frame["donor_distance"].astype(str)
        all_blocks = pd.concat(
            [pd.read_parquet(outputs[model] / "block_metrics.parquet") for model in outputs],
            ignore_index=True,
        )
        study_config = (
            ScientificStudyConfig.bounded()
            if panel_manifest["mode"] == "bounded" else ScientificStudyConfig.full()
        )
        validate_study_grid(case_frame, combined, all_blocks, study_config)
    numeric = combined.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("refuse comparison: non-finite segment metric")
    args.output.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output / "segment_metrics.parquet", index=False)
    pd.concat([pd.read_parquet(outputs[model] / "token_metrics.parquet") for model in outputs]).to_parquet(
        args.output / "token_metrics.parquet", index=False)
    pd.concat([pd.read_parquet(outputs[model] / "block_metrics.parquet") for model in outputs]).to_parquet(
        args.output / "block_metrics.parquet", index=False)
    needle_combined = pd.concat(
        [pd.read_parquet(outputs[model] / "needle_metrics.parquet") for model in outputs],
        ignore_index=True,
    )
    needle_combined.to_parquet(args.output / "needle_metrics.parquet", index=False)
    needle_cases: list[dict[str, object]] = []
    for keys, values in needle_combined.groupby(["model", "case_id"]):
        controls = values.set_index("control")
        carried, reset = controls.loc["carried"], controls.loc["reset"]
        needle_cases.append({
            "model": keys[0], "case_id": keys[1], "host_accession": carried["host_accession"],
            "distance_segments": int(carried["distance_segments"]),
            "distractor_count": int(carried["distractor_count"]),
            "carried_minus_reset_log_probability": float(
                carried["target_log_probability"] - reset["target_log_probability"]
            ),
            "recall_at_1": bool(carried["recall_at_1"]), "recall_at_5": bool(carried["recall_at_5"]),
            "exact_two_token_recovery": bool(carried["exact_two_token_recovery"]),
            "needle_absent_false_recall": bool(controls.loc["needle_absent", "recall_at_1"]),
            "incorrect_value_false_recall": bool(controls.loc["incorrect_value", "recall_at_1"]),
        })
    needle_case_frame = pd.DataFrame(needle_cases)
    needle_case_frame.to_parquet(args.output / "needle_case_metrics.parquet", index=False)
    detector_frame = _detector_frame(combined)
    detector_frame.to_parquet(args.output / "case_segment_features.parquet", index=False)
    _atomic_json(args.output / "correlations.json", _correlation_payload(detector_frame))
    prediction_tables: list[pd.DataFrame] = []
    coefficient_tables: list[pd.DataFrame] = []
    performance_rows: list[dict[str, object]] = []
    if detector_frame["host_accession"].nunique() >= 3:
        for model in models:
            model_frame = detector_frame[detector_frame["model"].eq(model)].copy()
            model_frame["study_row_id"] = np.arange(len(model_frame))
            model_frame["memory_detector_logit"] = 0.0
            memory_predictions: pd.DataFrame | None = None
            for detector, features in (("reference", REFERENCE_FEATURES), ("memory", MEMORY_FEATURES)):
                predictions, coefficients = nested_leave_one_host_out(model_frame, features)
                predictions["detector"] = detector
                coefficients["detector"] = detector
                coefficients["model"] = model
                prediction_tables.append(predictions)
                coefficient_tables.append(coefficients)
                performance_rows.append(_performance(predictions, detector, model))
                if detector == "memory":
                    memory_predictions = predictions
            if memory_predictions is None:
                raise RuntimeError("memory detector did not produce out-of-fold predictions")
            memory_logit = np.log(
                np.clip(memory_predictions["prediction"], 1e-9, 1 - 1e-9)
                / np.clip(1 - memory_predictions["prediction"], 1e-9, 1)
            )
            score_lookup = dict(zip(memory_predictions["study_row_id"], memory_logit))
            model_frame["memory_detector_logit"] = model_frame["study_row_id"].map(score_lookup)
            predictions, coefficients = nested_leave_one_host_out(model_frame, COUNTERFACTUAL_FEATURES)
            predictions["detector"] = "counterfactual"
            coefficients["detector"] = "counterfactual"
            coefficients["model"] = model
            prediction_tables.append(predictions)
            coefficient_tables.append(coefficients)
            performance_rows.append(_performance(predictions, "counterfactual", model))
    predictions = pd.concat(prediction_tables, ignore_index=True) if prediction_tables else pd.DataFrame()
    coefficients = pd.concat(coefficient_tables, ignore_index=True) if coefficient_tables else pd.DataFrame()
    performance = pd.DataFrame(performance_rows)
    predictions.to_parquet(args.output / "out_of_fold_predictions.parquet", index=False)
    coefficients.to_parquet(args.output / "outer_fold_coefficients.parquet", index=False)
    performance.to_parquet(args.output / "host_level_detection_metrics.parquet", index=False)
    detector_frame["composite_anomaly_score"] = 0.0
    if not predictions.empty:
        memory_scores = predictions[predictions["detector"].eq("memory")]
        score_lookup = {
            (row.model, row.case_id, row.sequence_class, int(row.relative_segment)): float(row.prediction)
            for row in memory_scores.itertuples()
        }
        detector_frame["composite_anomaly_score"] = [
            score_lookup.get((row.model, row.case_id, row.sequence_class, int(row.relative_segment)), 0.0)
            for row in detector_frame.itertuples()
        ]
        detector_frame.to_parquet(args.output / "case_segment_features.parquet", index=False)
    _atomic_json(args.output / "composite_scores.json", {
        "detectors": analysis_contract()["detectors"],
        "out_of_fold_rows": len(predictions),
        "prediction_sha256": (
            hashlib.sha256(predictions.to_csv(index=False).encode()).hexdigest()
            if len(predictions) else None
        ),
    })

    host_performance_rows: list[dict[str, object]] = []
    if not predictions.empty:
        from sklearn.metrics import average_precision_score, roc_auc_score

        for (model, detector, host), values in predictions.groupby(["model", "detector", "outer_host"]):
            labels = values["is_foreign"].astype(int).to_numpy()
            scores = values["prediction"].to_numpy(dtype=float)
            threshold = float(values["training_native_threshold_1pct_fpr"].iloc[0])
            positives = labels.astype(bool)
            native = values["is_native_calibration"].astype(bool).to_numpy()
            host_performance_rows.append({
                "model": model, "detector": detector, "host_accession": host,
                "sample_unit": "host", "auprc": float(average_precision_score(labels, scores)),
                "auroc": float(roc_auc_score(labels, scores)),
                "tpr_at_1pct_fpr": float((scores[positives] >= threshold).mean()),
                "false_positives_per_mb": float(
                    (scores[native] >= threshold).sum() * 1_000_000 / values.loc[native, "bases"].sum()
                ),
            })
    host_performance = pd.DataFrame(host_performance_rows)
    host_performance.to_parquet(args.output / "host_fold_detection_metrics.parquet", index=False)
    localization_rows: list[dict[str, object]] = []
    if not predictions.empty:
        foreign_predictions = predictions[predictions["sequence_class"].str.startswith("foreign_")]
        for keys, values in foreign_predictions.groupby(["model", "detector", "case_id", "sequence_class"]):
            ordered = values.sort_values("relative_segment")
            truth = ordered["is_foreign"].to_numpy(dtype=bool)
            threshold = float(ordered["training_native_threshold_1pct_fpr"].iloc[0])
            called = ordered["prediction"].to_numpy(dtype=float) >= threshold
            intersection, union = int((truth & called).sum()), int((truth | called).sum())
            true_indices, called_indices = np.flatnonzero(truth), np.flatnonzero(called)
            boundary_error = (
                int(called_indices[np.argmin(np.abs(called_indices - true_indices[0]))] - true_indices[0])
                if len(called_indices) else len(called)
            )
            end = int(true_indices[-1] + 1)
            recovery = next(
                (index - end for index in range(end, len(called)) if not called[index]),
                len(called) - end + 1,
            )
            localization_rows.append({
                "model": keys[0], "detector": keys[1], "case_id": keys[2],
                "sequence_class": keys[3], "host_accession": ordered["host_accession"].iloc[0],
                "localization_iou": intersection / union if union else 0.0,
                "boundary_error_segments": boundary_error, "recovery_segments": recovery,
                "sample_unit": "case; aggregate to host before inference",
            })
    pd.DataFrame(localization_rows).to_parquet(args.output / "case_localization_metrics.parquet", index=False)

    statistical_rows: list[dict[str, object]] = []
    if not performance.empty:
        if len(models) == 2:
            for endpoint in ("auprc", "auroc", "tpr_at_1pct_fpr", "false_positives_per_mb"):
                if endpoint not in host_performance:
                    continue
                for detector, selected_detector in host_performance.groupby("detector"):
                    pivot = selected_detector.pivot(index="host_accession", columns="model", values=endpoint)
                    statistical_rows.append({
                        "hypothesis_family": "likelihood_composition", "endpoint": endpoint,
                        "contrast": f"C19-C16/{detector}", "sample_unit": "host",
                        **paired_effect((pivot["C19"] - pivot["C16"]).to_dict()),
                    })
        for model in models:
            selected = host_performance[host_performance["model"].eq(model)]
            for endpoint in ("auprc", "tpr_at_1pct_fpr"):
                pivot = selected.pivot(index="host_accession", columns="detector", values=endpoint)
                differences = (pivot["memory"] - pivot["reference"]).to_dict()
                statistical_rows.append({
                    "hypothesis_family": "likelihood_composition", "endpoint": endpoint,
                    "contrast": f"{model}/memory-reference", **paired_effect(differences),
                })
        localization = pd.DataFrame(localization_rows)
        if len(models) == 2:
            for endpoint in ("localization_iou", "boundary_error_segments", "recovery_segments"):
                host_values = localization.groupby(
                    ["model", "detector", "host_accession"], as_index=False
                )[endpoint].median()
                for detector, selected_detector in host_values.groupby("detector"):
                    pivot = selected_detector.pivot(index="host_accession", columns="model", values=endpoint)
                    statistical_rows.append({
                        "hypothesis_family": "likelihood_composition", "endpoint": endpoint,
                        "contrast": f"C19-C16/{detector}",
                        **paired_effect((pivot["C19"] - pivot["C16"]).to_dict()),
                    })
            for endpoint in (
                "carried_minus_reset_log_probability", "recall_at_1", "exact_two_token_recovery"
            ):
                host_values = needle_case_frame.groupby(["model", "host_accession"], as_index=False)[endpoint].mean()
                pivot = host_values.pivot(index="host_accession", columns="model", values=endpoint)
                statistical_rows.append({
                    "hypothesis_family": "needle_metrics", "endpoint": endpoint,
                    "contrast": "C19-C16", **paired_effect((pivot["C19"] - pivot["C16"]).to_dict()),
                })
    # Predeclared host-level region contrasts for likelihood and every aggregate
    # memory parameter. Cases are reduced to host effects before inference.
    feature_families = {
        "likelihood_composition": [
            "bpb", "context_conflict", "gc_deviation", "markov_nll",
            "tetranucleotide_distance", *[f"jsd_k{k}" for k in range(1, 7)],
        ],
        "surprise_update_drift": [
            "surprise_norm", "surprise_norm_diff", "momentary_surprise_rms_max",
            "past_surprise_rms_max", "combined_surprise_rms_max", "memory_update_norm",
            "state_drift_norm", "state_drift_norm_diff", "raw_gradient_rms_max",
            "conditioned_gradient_rms_max", "gradient_scale_min",
            "gradient_intervention_fraction", "forgotten_weight_rms_max",
            "past_momentary_cosine_mean",
        ],
        "retrieval_gates": [
            "retrieval_norm", *[
                f"{gate}_{stat}" for gate in ("alpha", "eta", "theta")
                for stat in ("mean", "std", "min", "max")
            ],
        ],
    }
    model_effects: dict[tuple[str, str, str], dict[str, float]] = {}
    for family, features in feature_families.items():
        for model in models:
            model_values = detector_frame[detector_frame["model"].eq(model)]
            for feature in features:
                case_effects: dict[str, dict[str, list[float]]] = {}
                for (case_id, host), foreign in model_values[
                    model_values["sequence_class"].str.startswith("foreign_")
                ].groupby(["case_id", "host_accession"]):
                    inside = foreign[foreign["is_foreign"]][feature]
                    pre = foreign[foreign["relative_segment"].between(-8, -1)][feature]
                    same = model_values[
                        model_values["case_id"].eq(case_id)
                        & model_values["sequence_class"].eq("same_host")
                        & model_values["relative_segment"].ge(0)
                        & model_values["relative_segment"].lt(foreign["length_segments"].iloc[0])
                    ][feature]
                    untouched = model_values[
                        model_values["case_id"].eq(case_id)
                        & model_values["sequence_class"].eq("untouched")
                        & model_values["relative_segment"].ge(0)
                        & model_values["relative_segment"].lt(foreign["length_segments"].iloc[0])
                    ][feature]
                    if not all(len(values) for values in (inside, pre, same, untouched)):
                        continue
                    host_values = case_effects.setdefault(str(host), {
                        "insert-pre": [], "insert-same_host": [], "insert-untouched": []
                    })
                    host_values["insert-pre"].append(float(inside.median() - pre.median()))
                    host_values["insert-same_host"].append(float(inside.median() - same.median()))
                    host_values["insert-untouched"].append(float(inside.median() - untouched.median()))
                for contrast in ("insert-pre", "insert-same_host", "insert-untouched"):
                    effects = {
                        host: float(np.mean(values[contrast])) for host, values in case_effects.items()
                    }
                    if effects:
                        model_effects[(model, feature, contrast)] = effects
                        statistical_rows.append({
                            "hypothesis_family": family, "endpoint": feature,
                            "contrast": f"{model}/{contrast}", **paired_effect(effects),
                        })
                for contrast in ("insert-pre", "insert-same_host", "insert-untouched"):
                    c16 = model_effects.get(("C16", feature, contrast))
                    c19 = model_effects.get(("C19", feature, contrast))
                    if len(models) == 2 and model == "C16" and c16 and c19:
                        differences = {host: c19[host] - c16[host] for host in sorted(set(c16) & set(c19))}
                        statistical_rows.append({
                            "hypothesis_family": family, "endpoint": feature,
                            "contrast": f"C19-C16/{contrast}", **paired_effect(differences),
                        })
    statistics = pd.DataFrame(statistical_rows)
    if not statistics.empty:
        statistics["adjusted_p_value"] = statistics.groupby("hypothesis_family")["p_value"].transform(
            lambda values: holm_adjust(values.tolist())
        )
    statistics.to_parquet(args.output / "statistical_tests.parquet", index=False)

    peaks: list[dict[str, object]] = []
    cross_correlations: list[dict[str, object]] = []
    for keys, values in detector_frame[
        detector_frame["sequence_class"].str.startswith("foreign_")
    ].groupby(["model", "case_id", "sequence_class"]):
        ordered = values.sort_values("relative_segment")
        mask = ordered["is_foreign"].to_numpy(dtype=bool)
        if mask.any() and (~mask).any():
            for feature in (
                "momentary_surprise_rms_max", "bpb", "context_conflict",
                "memory_update_norm", "composite_anomaly_score",
            ):
                result = peak_enrichment(ordered[feature], mask)
                result["peak_latency_segments"] = int(
                    ordered.iloc[int(result["global_peak_index"])]["relative_segment"]
                )
                native_values = detector_frame[
                    detector_frame["model"].eq(keys[0])
                    & detector_frame["sequence_class"].eq("native_calibration")
                ][feature]
                native_95 = float(np.quantile(native_values, 0.95))
                length = int(ordered["length_segments"].iloc[0])
                post = ordered[ordered["relative_segment"].ge(length)]
                recovered = post.loc[post[feature].lt(native_95), "relative_segment"]
                result["recovery_to_native95_segments"] = (
                    int(recovered.iloc[0] - length) if len(recovered) else len(post) + 1
                )
                peaks.append({
                    "model": keys[0], "case_id": keys[1], "sequence_class": keys[2],
                    "feature": feature, **result,
                })
            for lag, correlation in lagged_spearman(
                ordered["bpb"], ordered["momentary_surprise_rms_max"]
            ).items():
                cross_correlations.append({
                    "model": keys[0], "case_id": keys[1], "sequence_class": keys[2],
                    "lag_segments": lag,
                    "spearman": correlation if math.isfinite(correlation) else 0.0,
                    "correlation_defined": math.isfinite(correlation),
                })
    pd.DataFrame(peaks).to_parquet(args.output / "surprise_peak_enrichment.parquet", index=False)
    pd.DataFrame(cross_correlations).to_parquet(args.output / "bpb_surprise_lagged_correlation.parquet", index=False)
    block_metrics = pd.read_parquet(args.output / "block_metrics.parquet")
    _write_core_figures(args.output, detector_frame, performance, block_metrics, needle_case_frame)
    _write_dictionary(args.output, detector_frame)
    telemetry_claim = "not evaluated in two-host smoke mode"
    if not statistics.empty:
        required = statistics[
            statistics["contrast"].astype(str).str.contains("memory-reference")
            & statistics["endpoint"].isin(["auprc", "tpr_at_1pct_fpr"])
        ]
        telemetry_claim = (
            "supported under the preregistered rule"
            if len(required) == 2 * len(models) and required["lower_95"].gt(0).all()
            else "not supported under the preregistered positive-lower-bound rule"
        )
    performance_lines = "\n".join(
        f"- {row.model} {row.detector}: AUPRC={row.auprc:.4f}, AUROC={row.auroc:.4f}, "
        f"TPR@1%FPR={row.tpr_at_1pct_fpr:.4f}, FP/Mb={row.false_positives_per_mb:.3f}, "
        f"Brier={row.brier_score:.4f}"
        for row in performance.itertuples()
    ) or "- Smoke mode: predictive models are deferred because two hosts cannot support nested host validation."
    needle_summary = needle_case_frame.groupby("model").agg(
        log_probability_gain=("carried_minus_reset_log_probability", "mean"),
        recall_at_1=("recall_at_1", "mean"),
        exact_recovery=("exact_two_token_recovery", "mean"),
    )
    needle_lines = "\n".join(
        f"- {model}: carried−reset log probability={row.log_probability_gain:.4f}, "
        f"Recall@1={row.recall_at_1:.4f}, exact recovery={row.exact_recovery:.4f}"
        for model, row in needle_summary.iterrows()
    )
    report_title = (
        "C16–C19 anomaly and DNA-needle validation study"
        if len(models) == 2 else "C19 anomaly and DNA-needle validation study"
    )
    scope = (
        "This bundle compares C19 and C16 only on byte-identical synthetic validation cases. "
        "Their contrast is combined checkpoint/exposure progression, not pure scaling."
        if len(models) == 2 else
        "This is the primary single-model C19 analysis. C16 was not required or loaded; "
        "a paired comparison can be added later from the same frozen panel."
    )
    report = f"""# {report_title}

## Scope

{scope} The protected test panel was never available to this workflow.

## Predictive conclusion

Telemetry utility is **{telemetry_claim}**. This rule requires positive lower 95% paired-host bootstrap bounds for both memory-minus-reference AUPRC and TPR at 1% FPR. Counterfactual performance is reported separately.

{performance_lines}

## Associative needle endpoints

{needle_lines}

## Exploratory association and mechanistic diagnostics

Host-bootstrap correlations, partial rank correlations, boundary-aligned effects, per-block normalized-depth telemetry, circular-shift peak enrichment, and BPB–surprise lags are retained in the machine-readable tables. Correlations are mechanistic diagnostics, not causal evidence.

## Interpretation contract

Exploratory associations, predictive performance, and mechanistic diagnostics must be reported separately. Surprise peaks can support anomaly sensitivity; they cannot establish horizontal transfer, biological function, pathogenicity, or causal adaptive-memory superiority. No final biological detector is claimed without an external locked test panel.

## Bibliography

- Vernikos and Parkhill, AlienHunter/IVOM, https://doi.org/10.1093/bioinformatics/btl369
- Langille et al., artificial-genome evaluation, https://pmc.ncbi.nlm.nih.gov/articles/PMC2760805/
- Bertelli et al., INSIDER, https://pmc.ncbi.nlm.nih.gov/articles/PMC8273350/
"""
    (args.output / "SCIENTIFIC_REPORT.md").write_text(report)
    _atomic_json(args.output / "analysis_contract.json", analysis_contract())
    (args.output / "analysis.log").write_text(
        f"completed={datetime.now(timezone.utc).isoformat()}\n"
        f"panel_contract_sha256={manifests[0]['panel_contract_sha256']}\n"
    )
    _atomic_json(args.output / "COMPLETE.json", {
        "panel_contract_sha256": manifests[0]["panel_contract_sha256"],
        "models": list(models),
        "comparison_label": (
            "combined checkpoint/exposure progression" if len(models) == 2 else "C19 single-model"
        ),
    })
    catalog = {
        str(path.relative_to(args.output)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(args.output.rglob("*")) if path.is_file()
    }
    _atomic_json(args.output / "catalog.json", catalog)
    archive = shutil.make_archive(str(args.output), "zip", root_dir=args.output)
    print(archive)
    return args.output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    contract = sub.add_parser("contract")
    contract.add_argument("--output", type=Path, required=True)
    panel = sub.add_parser("freeze")
    panel.add_argument("--dataset-dir", type=Path, required=True)
    panel.add_argument("--validation-panel", type=Path, required=True)
    panel.add_argument("--e25-panel", type=Path, required=True)
    panel.add_argument("--ani-pairs", type=Path, required=True)
    panel.add_argument("--ani-membership", type=Path, required=True)
    panel.add_argument("--mode", choices=("smoke", "bounded", "full"), default="smoke")
    panel.add_argument("--output", type=Path, required=True)
    estimate = sub.add_parser("estimate-runtime")
    estimate.add_argument("--checkpoint", type=Path, required=True)
    estimate.add_argument("--frozen-panel", type=Path, required=True)
    estimate.add_argument("--output", type=Path, required=True)
    estimate.add_argument("--device", default="auto")
    estimate.add_argument("--probe-forwards", type=int, default=64)
    estimate.add_argument("--max-hours", type=float, default=22.0)
    estimate.add_argument("--trust-owned-checkpoint", action="store_true")
    run = sub.add_parser("run-model")
    run.add_argument("--model", choices=("C16", "C19"), required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--dataset-dir", type=Path, required=True)
    run.add_argument("--validation-panel", type=Path, required=True)
    run.add_argument("--frozen-panel", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="auto")
    run.add_argument("--max-runtime-hours", type=float)
    run.add_argument("--trust-owned-checkpoint", action="store_true")
    analysis = sub.add_parser("compare")
    analysis.add_argument("--input", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "contract":
        _atomic_json(args.output, analysis_contract())
        print(args.output)
    elif args.command == "freeze":
        print(freeze(args))
    elif args.command == "estimate-runtime":
        print(estimate_runtime(args))
    elif args.command == "run-model":
        print(run_model(args))
    else:
        print(compare(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
