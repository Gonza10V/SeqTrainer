"""Run immutable Stage C context-anomaly and DNA-needle evaluations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from seqtrainer.data.bacteria_titan import StageCPanelManifest, TokenStreamDataset, validate_panel_against_dataset

from .checkpoints import checkpoint_parent_dataset_fingerprint
from .config import MemoryMode, StageCModelConfig
from .context_eval import (
    BENCHMARK_VERSION,
    INTERVENTIONS,
    SEQUENCE_CLASSES,
    CaseResumeStore,
    ContextEvalConfig,
    TokenStreamSlice,
    contract_hash,
    boundary_metrics,
    detection_metrics,
    materialize_anomaly_sequences,
    materialize_needle_sequence,
    needle_metrics,
    paired_host_bootstrap,
    regenerate_catalog,
    select_anomaly_cases,
    select_needle_cases,
    sha256_bytes,
    sha256_file,
    stage_checkpoint,
    write_jsonl,
    write_sequence_slices,
)
from .model import BlockStates, StageCPaperMACForCausalLM, detach_stream_states
from .tokenizers import SeqTrainerBaseTokenizer, SixMerTokenizer


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _load_checkpoint(path: Path, device: torch.device, *, trusted: bool) -> Mapping[str, object]:
    if not trusted:
        raise ValueError("full-state checkpoint evaluation requires --trust-owned-checkpoint")
    kwargs: dict[str, object] = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    payload = torch.load(path, **kwargs)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_config"), Mapping):
        raise ValueError("invalid Stage C checkpoint")
    return payload


def _tokenizer(name: str):
    if name == "nonoverlap_6mer_v1":
        return SixMerTokenizer()
    if name == "seqtrainer_base_v1":
        return SeqTrainerBaseTokenizer()
    raise ValueError(f"context evaluation does not have a lossless decoder for {name!r}")


def _slices(dataset: TokenStreamDataset, panel: StageCPanelManifest, split: str, decode) -> dict[str, TokenStreamSlice]:
    streams = dataset.streams(split=split, stream_ids=panel.stream_ids)
    result: dict[str, TokenStreamSlice] = {}
    for item in dataset.index:
        if item.stream_id not in streams:
            continue
        start, end = item.token_offset, item.token_offset + item.token_count
        tokens = tuple(int(value) for value in dataset.tokens[item.shard_index][start:end])
        lengths = tuple(int(value) for value in dataset.base_lengths[item.shard_index][start:end])
        result[item.stream_id] = TokenStreamSlice(
            item.stream_id, item.accession, item.clade_group, tokens, lengths, decode(tokens)
        )
    return result


def _warm(
    model: StageCPaperMACForCausalLM,
    tokens: Sequence[int],
    segments: int,
    *,
    stream_id: str,
    device: torch.device,
) -> BlockStates:
    states = model.initial_states(stream_id)
    for index in range(segments):
        inputs = torch.tensor(tokens[index * 32 : (index + 1) * 32], device=device).long()
        output = model.forward_segment((states,), inputs, memory_mode=MemoryMode.ADAPTIVE)
        states = detach_stream_states(output.states[0])
    return states


def _score_segments(
    model: StageCPaperMACForCausalLM,
    tokens: Sequence[int],
    *,
    start_segment: int,
    initial_states: BlockStates,
    mode: MemoryMode,
    decode,
    device: torch.device,
) -> list[dict[str, object]]:
    states = initial_states
    rows: list[dict[str, object]] = []
    segment_count = (len(tokens) - 1) // 32
    for segment in range(start_segment, segment_count):
        start = segment * 32
        inputs = torch.tensor(tokens[start : start + 32], device=device).long()
        labels = torch.tensor(tokens[start + 1 : start + 33], device=device).long()
        output = model.forward_segment((states,), inputs, memory_mode=mode)
        token_nll = F.cross_entropy(output.logits[0], labels, reduction="none").detach().float().cpu().numpy()
        bases = np.asarray([len(decode((int(value),))) for value in labels], dtype=np.int64)
        nll = float(token_nll.sum())
        rows.append({
            "segment_index": segment,
            "nll_nats": nll,
            "tokens": 32,
            "bases": int(bases.sum()),
            "bpb": nll / (math.log(2.0) * int(bases.sum())),
            "token_nll": token_nll.tolist(),
            "token_bpb": (token_nll / (math.log(2.0) * bases)).tolist(),
            "retrieval_norm": output.retrieval_norm,
            "memory_update_norm": output.memory_update_norm,
            "surprise_norm": output.surprise_norm,
            "state_drift_norm": output.state_drift_norm,
            **output.gate_statistics,
            **output.memory_gradient_statistics,
            "finite": bool(
                math.isfinite(nll)
                and math.isfinite(output.retrieval_norm)
                and math.isfinite(output.memory_update_norm)
                and math.isfinite(output.surprise_norm)
                and math.isfinite(output.state_drift_norm)
                and all(math.isfinite(float(value)) for value in output.gate_statistics.values())
                and all(math.isfinite(float(value)) for value in output.memory_gradient_statistics.values())
            ),
            "block_diagnostics": list(output.block_diagnostics),
        })
        states = detach_stream_states(output.states[0])
    return rows


def _run_anomaly_case(
    case, streams, model, decode, device
) -> list[dict[str, object]]:
    sequences = materialize_anomaly_sequences(case, streams)
    host = streams[case.host_stream_id]
    wrong = streams[case.wrong_host_stream_id]
    carried = _warm(model, host.token_ids, case.insertion_segment, stream_id=case.host_stream_id, device=device)
    wrong_state = _warm(model, wrong.token_ids, case.insertion_segment, stream_id=wrong.stream_id, device=device)
    rows: list[dict[str, object]] = []
    for sequence_class in SEQUENCE_CLASSES:
        sequence = sequences[sequence_class]
        for intervention in INTERVENTIONS:
            if intervention == "carried":
                initial, mode = carried, MemoryMode.ADAPTIVE
            elif intervention == "wrong_host":
                initial, mode = wrong_state, MemoryMode.ADAPTIVE
            elif intervention == "reset":
                initial, mode = model.initial_states(case.host_stream_id + ":reset"), MemoryMode.ADAPTIVE
            else:
                initial, mode = model.initial_states(case.host_stream_id + ":none"), MemoryMode.NONE
            for row in _score_segments(
                model, sequence, start_segment=case.insertion_segment,
                initial_states=initial, mode=mode, decode=decode, device=device,
            ):
                rows.append({
                    "benchmark": "anomaly", "case_id": case.case_id,
                    "host_accession": case.host_accession,
                    "sequence_class": sequence_class, "intervention": intervention,
                    "sequence_sha256": case.sequence_sha256[sequence_class],
                    "relative_segment": int(row["segment_index"]) - case.insertion_segment,
                    **row,
                })
    return rows


def _log_probability(logits: torch.Tensor, targets: Sequence[int], positions: Sequence[int]) -> tuple[float, int, float, bool]:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    values = [float(log_probs[position, int(target)].detach().cpu()) for position, target in zip(positions, targets)]
    first = log_probs[positions[0]]
    rank = int((first > first[int(targets[0])]).sum().item()) + 1
    reciprocal = 1.0 / rank
    exact = all(int(logits[position].argmax().item()) == int(target) for position, target in zip(positions, targets))
    return sum(values), rank, reciprocal, exact


def _run_needle_case(
    case, streams, model, device, *, sequence_tokens=None, wrong_host_tokens=None
) -> list[dict[str, object]]:
    sequence = (
        tuple(map(int, sequence_tokens))
        if sequence_tokens is not None
        else materialize_needle_sequence(case, streams)
    )
    wrong = streams[case.wrong_host_stream_id]
    wrong_tokens = wrong.token_ids if wrong_host_tokens is None else tuple(map(int, wrong_host_tokens))
    carried = _warm(model, sequence, case.query_segment, stream_id=case.host_stream_id, device=device)
    wrong_state = _warm(model, wrong_tokens, case.query_segment, stream_id=wrong.stream_id, device=device)
    write = case.write_segment * 32
    absent_sequence = list(sequence)
    replacement = (int(absent_sequence[write]) + 1) % model.config.vocab_size
    if replacement == model.config.pad_token_id:
        replacement = (replacement + 1) % model.config.vocab_size
    absent_sequence[write] = replacement
    incorrect_sequence = list(sequence)
    replacement = (int(incorrect_sequence[write + 4]) + 1) % model.config.vocab_size
    if replacement == model.config.pad_token_id:
        replacement = (replacement + 1) % model.config.vocab_size
    incorrect_sequence[write + 4] = replacement
    absent_state = _warm(model, absent_sequence, case.query_segment, stream_id=case.host_stream_id + ":absent", device=device)
    incorrect_state = _warm(model, incorrect_sequence, case.query_segment, stream_id=case.host_stream_id + ":incorrect", device=device)
    query = case.query_segment * 32
    inputs = torch.tensor(sequence[query : query + 32], device=device).long()
    target_positions = (3, 4)  # logits at final key token and first value token
    controls = ("carried", "reset", "wrong_host", "needle_absent", "incorrect_value", "no_memory")
    rows: list[dict[str, object]] = []
    for control in controls:
        control_inputs = inputs.clone()
        targets = case.value_tokens
        if control == "carried":
            state, mode = carried, MemoryMode.ADAPTIVE
        elif control == "wrong_host":
            state, mode = wrong_state, MemoryMode.ADAPTIVE
        elif control == "no_memory":
            state, mode = model.initial_states(case.host_stream_id + ":none"), MemoryMode.NONE
        elif control == "needle_absent":
            state, mode = absent_state, MemoryMode.ADAPTIVE
        elif control == "incorrect_value":
            state, mode = incorrect_state, MemoryMode.ADAPTIVE
        else:
            state, mode = model.initial_states(case.host_stream_id + ":reset"), MemoryMode.ADAPTIVE
        output = model.forward_segment((state,), control_inputs, memory_mode=mode)
        probability, rank, reciprocal, exact = _log_probability(output.logits[0], targets, target_positions)
        rows.append({
            "benchmark": "needle", "case_id": case.case_id,
            "host_accession": case.host_accession,
            "distance_segments": case.distance_segments,
            "distractor_count": case.distractor_count, "control": control,
            "sequence_sha256": case.sequence_sha256,
            "target_log_probability": probability, "first_token_rank": rank,
            "recall_at_1": rank <= 1, "recall_at_5": rank <= 5,
            "reciprocal_rank": reciprocal, "exact_two_token_recovery": exact,
        })
    return rows


def _anomaly_summary(rows: Sequence[Mapping[str, object]], case_lengths: Mapping[str, int], thresholds=None) -> dict[str, object]:
    carried = [row for row in rows if row["intervention"] == "carried"]
    reset_lookup = {
        (row["case_id"], row["sequence_class"], row["relative_segment"]): float(row["bpb"])
        for row in rows if row["intervention"] == "reset"
    }
    labels = [int(row["sequence_class"] == "different_ani" and int(row["relative_segment"]) < case_lengths[str(row["case_id"])]) for row in carried]
    scores = [float(row["bpb"]) - reset_lookup[(row["case_id"], row["sequence_class"], row["relative_segment"])] for row in carried]
    bases = [int(row["bases"]) for row in carried]
    summary = detection_metrics(labels, scores, thresholds=thresholds, represented_bases=bases)
    summary["score"] = "context_conflict = BPB_carried - BPB_reset"
    threshold = float(summary["thresholds"]["fpr_0.01"])
    per_case: dict[str, object] = {}
    for case_id, length in sorted(case_lengths.items()):
        case_scores = [
            scores[index] for index, row in enumerate(carried)
            if row["case_id"] == case_id and row["sequence_class"] == "different_ani"
        ]
        per_case[case_id] = boundary_metrics(
            case_scores, boundary=0, threshold=threshold, positive_end=length
        )
    summary["per_case_boundary"] = per_case
    for key in ("boundary_error", "detection_delay", "recovery_time"):
        values = [float(value[key]) for value in per_case.values() if value[key] is not None]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None
    host_conflicts: dict[str, list[float]] = {}
    for index, row in enumerate(carried):
        if labels[index]:
            host_conflicts.setdefault(str(row["host_accession"]), []).append(scores[index])
    summary["paired_host_bootstrap"] = paired_host_bootstrap(host_conflicts)
    for diagnostic in ("retrieval_norm", "memory_update_norm", "surprise_norm", "state_drift_norm"):
        reset_values = {
            (row["case_id"], row["sequence_class"], row["relative_segment"]): float(row[diagnostic])
            for row in rows if row["intervention"] == "reset"
        }
        deltas = [
            float(row[diagnostic]) - reset_values[(row["case_id"], row["sequence_class"], row["relative_segment"])]
            for row in carried
        ]
        summary[f"carried_minus_reset_{diagnostic}_mean"] = float(np.mean(deltas))
    return summary


def _write_report(output: Path, manifest: Mapping[str, object], anomaly: Mapping[str, object], needle: Mapping[str, object]) -> None:
    lines = [
        "# Stage C context-anomaly and needle evaluation", "",
        f"- Model: `{manifest['model_id']}`", f"- Split: `{manifest['split']}`",
        f"- Checkpoint SHA-256: `{manifest['checkpoint_sha256']}`",
        f"- Benchmark: `{BENCHMARK_VERSION}`", "", "## Cross-ANI anomaly", "",
        f"- AUPRC: `{float(anomaly['auprc']):.6f}`",
        f"- TPR at 1% FPR: `{float(anomaly['tpr_at_0.01']):.6f}`",
        f"- TPR at 0.1% FPR: `{float(anomaly['tpr_at_0.001']):.6f}`", "",
        f"- False positives/Mb at 1% FPR: `{float(anomaly['false_positives_per_mb_at_0.01']):.6f}`",
        f"- Mean boundary error (segments): `{anomaly['mean_boundary_error']}`",
        f"- Mean detection delay (segments): `{anomaly['mean_detection_delay']}`",
        f"- Mean recovery time (segments): `{anomaly['mean_recovery_time']}`", "",
        "## Needle", "",
        f"- Carried-reset mean log-probability gain: `{float(needle['carried_minus_reset']['mean']):.6f}`",
        f"- First-token mean rank: `{float(needle['first_token_rank']):.6f}`",
        f"- Recall@1 / Recall@5: `{float(needle['recall_at_1']):.6f}` / `{float(needle['recall_at_5']):.6f}`",
        f"- Exact two-token recovery: `{float(needle['exact_two_token_recovery']):.6f}`",
        f"- Promising long-context signal: `{needle['promising_long_context_signal']}`", "",
        "These are inference interventions on one checkpoint, not separately trained matched controls.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    checkpoint = args.model_dir / "checkpoint.pt"
    model_manifest = json.loads((args.model_dir / "model_manifest.json").read_text(encoding="utf-8"))
    payload = _load_checkpoint(checkpoint, device, trusted=args.trust_owned_checkpoint)
    model_config = StageCModelConfig.from_dict(payload["model_config"])
    tokenizer = _tokenizer(model_config.tokenizer_name)
    model = StageCPaperMACForCausalLM(model_config).to(device).eval()
    model.load_state_dict(payload["model_state"])
    dataset = TokenStreamDataset(args.dataset_dir, verify_checksums=True)
    panel = StageCPanelManifest.from_path(args.panel_manifest)
    validate_panel_against_dataset(panel, dataset)
    dataset_hash = sha256_file(args.dataset_dir / "token_stream_manifest.json")
    checkpoint_dataset = checkpoint_parent_dataset_fingerprint(payload)
    if checkpoint_dataset != dataset_hash:
        raise ValueError("checkpoint and ordered token dataset fingerprints disagree")
    if panel.payload["split"] != args.split:
        raise ValueError("panel and requested split disagree")
    frozen_thresholds = None
    if args.split == "test":
        if not args.run_locked_test or args.validation_bundle is None:
            raise ValueError("test evaluation requires --run-locked-test and --validation-bundle")
        if args.mode is not None:
            raise ValueError("locked test refuses --mode; validation parameters are copied verbatim")
        validation_manifest = json.loads((args.validation_bundle / "evaluation_manifest.json").read_text(encoding="utf-8"))
        validation_summary = json.loads((args.validation_bundle / "anomaly_summary.json").read_text(encoding="utf-8"))
        if validation_manifest.get("split") != "val" or not (args.validation_bundle / "COMPLETE.json").is_file():
            raise ValueError("locked test requires a completed validation bundle")
        if validation_manifest.get("benchmark_version") != BENCHMARK_VERSION:
            raise ValueError("validation bundle benchmark version changed")
        if validation_manifest.get("model_id") != model_manifest["model_id"]:
            raise ValueError("validation bundle belongs to a different model")
        if validation_manifest.get("dataset_fingerprint") != dataset_hash:
            raise ValueError("validation bundle belongs to a different dataset")
        if validation_manifest.get("parameters", {}).get("mode") != "full":
            raise ValueError("locked test requires a completed full validation bundle")
        config = ContextEvalConfig(**{**validation_manifest["parameters"], "split": "test"})
        frozen_thresholds = validation_summary["thresholds"]
    else:
        config = ContextEvalConfig.full(split="val") if args.mode == "full" else ContextEvalConfig()
        if args.mode == "full":
            smoke_complete = False
            base = args.registry / "evaluations" / BENCHMARK_VERSION / str(model_manifest["model_id"]) / "val"
            for complete in base.glob("*/COMPLETE.json"):
                candidate_path = complete.parent / "evaluation_manifest.json"
                if not candidate_path.is_file():
                    continue
                candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
                parameters = candidate.get("parameters", {})
                if (
                    parameters.get("mode") == "smoke"
                    and candidate.get("dataset_fingerprint") == dataset_hash
                    and candidate.get("panel_sha256") == panel.hash
                ):
                    smoke_complete = True
                    break
            if not smoke_complete:
                raise ValueError("full validation requires a completed matching smoke bundle")
    slices = _slices(dataset, panel, args.split, tokenizer.decode)
    anomaly_cases = select_anomaly_cases(tuple(slices.values()), config)
    needle_cases = select_needle_cases(tuple(slices.values()), config)
    case_hash = contract_hash([*[case.to_dict() for case in anomaly_cases], *[case.to_dict() for case in needle_cases]])
    manifest = {
        "format_version": 1, "benchmark_version": BENCHMARK_VERSION,
        "model_id": model_manifest["model_id"], "checkpoint_sha256": model_manifest["checkpoint_sha256"],
        "dataset_fingerprint": dataset_hash, "panel_sha256": panel.hash, "split": args.split,
        "seed": config.seed, "case_list_sha256": case_hash, "code_commit": args.code_commit,
        "parameters": config.to_dict(), "validation_bundle": str(args.validation_bundle) if args.validation_bundle else None,
        "locked_validation_thresholds": frozen_thresholds,
        "interventions": list(INTERVENTIONS), "sequence_classes": list(SEQUENCE_CLASSES),
        "anomaly_score": "bpb_carried_minus_bpb_reset",
        "needle_primary": "summed_target_value_log_probability_carried_minus_reset",
    }
    manifest["config_sha256"] = contract_hash(manifest)
    output = args.output_dir or args.registry / "evaluations" / BENCHMARK_VERSION / str(manifest["model_id"]) / args.split / str(manifest["config_sha256"])[:12]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation_manifest.json").exists():
        if json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("evaluation manifest changed; refuse overwrite")
    else:
        _atomic_json(output / "evaluation_manifest.json", manifest)
    write_jsonl(output / "anomaly_cases.jsonl", (case.to_dict() for case in anomaly_cases))
    write_jsonl(output / "needle_cases.jsonl", (case.to_dict() for case in needle_cases))
    retained: dict[str, str] = {}
    for case in anomaly_cases:
        host, donor = slices[case.host_stream_id], slices[case.donor_stream_id]
        retained[f"{case.case_id}|host_window"] = host.base_block(
            case.host_window_start_segment,
            case.host_window_end_segment - case.host_window_start_segment,
        )
        retained[f"{case.case_id}|donor_fragment"] = donor.base_block(
            case.donor_start_segment, case.length_segments
        )
        retained[f"{case.case_id}|hard_negative"] = host.base_block(
            case.hard_negative_start_segment, case.length_segments
        )
        wrong = slices[case.wrong_host_stream_id]
        retained[f"{case.case_id}|wrong_host_warmup"] = wrong.base_block(
            0, case.insertion_segment
        )
        for label, expected in case.retained_slice_sha256.items():
            actual = sha256_bytes(retained[f"{case.case_id}|{label}"].encode("ascii"))
            if actual != expected:
                raise ValueError(f"retained anomaly slice changed: {case.case_id}/{label}")
    for case in needle_cases:
        sequence = materialize_needle_sequence(case, slices)
        retained[f"{case.case_id}|needle_window"] = tokenizer.decode(sequence)
        wrong = slices[case.wrong_host_stream_id]
        retained[f"{case.case_id}|wrong_host_warmup"] = wrong.base_block(
            0, case.query_segment
        )
        for label, expected in case.retained_slice_sha256.items():
            actual = sha256_bytes(retained[f"{case.case_id}|{label}"].encode("ascii"))
            if actual != expected:
                raise ValueError(f"retained needle slice changed: {case.case_id}/{label}")
    write_sequence_slices(output / "sequence_slices.fasta.gz", sorted(retained.items()))
    store = CaseResumeStore(output / "resume", manifest)
    all_rows: list[dict[str, object]] = []
    try:
        for case in anomaly_cases:
            if not store.completed(case.case_id):
                store.commit(case.case_id, {"rows": _run_anomaly_case(case, slices, model, tokenizer.decode, device)})
            all_rows.extend(store.load(case.case_id)["result"]["rows"])
        needle_rows: list[dict[str, object]] = []
        for case in needle_cases:
            if not store.completed(case.case_id):
                store.commit(case.case_id, {"rows": _run_needle_case(case, slices, model, device)})
            rows = store.load(case.case_id)["result"]["rows"]
            all_rows.extend(rows)
            grouped = {str(row["control"]): row for row in rows}
            needle_rows.append({
                **case.to_dict(),
                "carried_log_probability": grouped["carried"]["target_log_probability"],
                "reset_log_probability": grouped["reset"]["target_log_probability"],
                **{
                    key: grouped["carried"][key]
                    for key in (
                        "first_token_rank", "recall_at_1", "recall_at_5",
                        "reciprocal_rank", "exact_two_token_recovery",
                    )
                },
            })
        # Persist diagnostic levels separately.  Block count is architecture
        # dependent, so the block table carries normalized depth and is never
        # treated as a table of biological replicates.
        block_rows: list[dict[str, object]] = []
        token_rows: list[dict[str, object]] = []
        flat_rows: list[dict[str, object]] = []
        for row in all_rows:
            flat = dict(row)
            diagnostics = flat.pop("block_diagnostics", [])
            token_nll = flat.pop("token_nll", None)
            token_bpb = flat.pop("token_bpb", None)
            flat_rows.append(flat)
            identity = {
                key: flat[key] for key in (
                    "benchmark", "case_id", "host_accession", "sequence_sha256"
                ) if key in flat
            }
            if "relative_segment" in flat:
                identity["relative_segment"] = flat["relative_segment"]
            if "intervention" in flat:
                identity["intervention"] = flat["intervention"]
            if "sequence_class" in flat:
                identity["sequence_class"] = flat["sequence_class"]
            for diagnostic in diagnostics:
                block_rows.append({**identity, **diagnostic})
            if token_nll is not None and token_bpb is not None:
                for position, (nll_value, bpb_value) in enumerate(zip(token_nll, token_bpb)):
                    token_rows.append({**identity, "token_position": position,
                                       "token_nll_nats": nll_value, "token_bpb": bpb_value})
        frame = pd.DataFrame(flat_rows)
        frame.to_parquet(output / "segment_scores.parquet", index=False)
        pd.DataFrame(token_rows).to_parquet(output / "token_scores.parquet", index=False)
        pd.DataFrame(block_rows).to_parquet(output / "block_scores.parquet", index=False)
        anomaly_summary = _anomaly_summary(
            [row for row in all_rows if row["benchmark"] == "anomaly"],
            {case.case_id: case.length_segments for case in anomaly_cases},
            frozen_thresholds,
        )
        needle_summary = needle_metrics(needle_rows, seed=config.seed, samples=config.bootstrap_samples)
        _atomic_json(output / "anomaly_summary.json", anomaly_summary)
        _atomic_json(output / "needle_summary.json", needle_summary)
        (output / "figures").mkdir(exist_ok=True)
        (output / "logs").mkdir(exist_ok=True)
        _write_report(output, manifest, anomaly_summary, needle_summary)
        (output / "FAILED.txt").unlink(missing_ok=True)
        _atomic_json(output / "COMPLETE.json", {"completed_at": datetime.now(timezone.utc).isoformat(), "manifest_sha256": contract_hash(manifest)})
        regenerate_catalog(args.registry)
    except BaseException as error:
        (output / "FAILED.txt").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage", help="stage an immutable checkpoint")
    stage.add_argument("--source", type=Path, required=True)
    stage.add_argument("--registry", type=Path, required=True)
    stage.add_argument("--metadata-json", type=Path, required=True)
    stage.add_argument("--trust-owned-checkpoint", action="store_true")
    evaluate = sub.add_parser("run", help="run smoke/full validation or locked test")
    evaluate.add_argument("--dataset-dir", type=Path, required=True)
    evaluate.add_argument("--panel-manifest", type=Path, required=True)
    evaluate.add_argument("--model-dir", type=Path, required=True)
    evaluate.add_argument("--registry", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument("--split", choices=("val", "test"), default="val")
    evaluate.add_argument("--mode", choices=("smoke", "full"))
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--code-commit", required=True)
    evaluate.add_argument("--trust-owned-checkpoint", action="store_true")
    evaluate.add_argument("--validation-bundle", type=Path)
    evaluate.add_argument("--run-locked-test", action="store_true")
    sub.add_parser("catalog").add_argument("--registry", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "stage":
        metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        print(stage_checkpoint(args.source, args.registry, metadata, trusted=args.trust_owned_checkpoint))
    elif args.command == "catalog":
        print(regenerate_catalog(args.registry))
    else:
        print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
