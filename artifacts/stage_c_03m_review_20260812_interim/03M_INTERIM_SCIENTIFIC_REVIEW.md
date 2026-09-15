# Stage C 03m interim scientific review — 12 August 2026

## Executive decision

**03m is running, not finished. E100 remains on HOLD.**

At the evidence boundary (2026-08-12 14:46 UTC), 03m was still in its first task: the uncapped held-out evaluation of the final C19 checkpoint. It had completed 18,816 of 225,220 planned segments (8.35%) and 3,247,372 valid bases. The observed rate was 0.474 segments/s after 11.02 hours, with an estimated 120.87 hours remaining for this task alone.

No final `evaluation.json`, controlled memory report, memory trace, generation evaluation, or scale-gate result exists. The notebook therefore has not produced an E25 scientific decision.

There is also an independent prerequisite failure: the configured c16 baseline directory contains only a capped pilot evaluation. The required `full_evaluation/evaluation.json` and `generation_t0p6/generation_evaluation.json` are absent. Consequently, even if the current C19 evaluation finishes, the final scale-gate command cannot complete until the full c16 baseline evaluation and matched generation run are produced.

The correct impact on the plan is unchanged: do not launch E100 and do not run 03r/03n as an E100 authorization. Preserve and resume the current evaluation, complete the missing baseline, and accept E100 only if the frozen gate eventually writes `scale_gate.json` with `proceed: true`.

## What 03m is designed to do

The notebook is a sequential five-stage workflow against the immutable final C19 checkpoint:

1. uncapped held-out evaluation on the eight-accession validation panel;
2. controlled memory behavior on 64 key/value pairs;
3. long-stream memory tracing (up to 12 streams and 512 segments);
4. matched generation at temperature 0.6; and
5. comparison with c16 under the frozen E25 prediction, memory, and generation thresholds.

Only stage 1 has started. Because the stages are sequential, absence of the later artifacts is expected while evaluation is running, but it means there are not yet results from which a complete scale decision can be made.

## Provenance and execution integrity

The running analysis has good provenance:

- evaluator commit: `ae72fae21ff9a0b50e4fe1d9d642c38c643b4923`;
- clean checkout recorded by the wrapper;
- device: NVIDIA A100-SXM4-40GB;
- checkpoint SHA-256 in the resume contract: `07fb2069b1f29a76898a90d8dfb899c5ca46cb90608fac45bc0ddff9876dbd1a`;
- dataset fingerprint: `2fbdb870606e6fb1ce0f6750524726d041474be5081da2cb2316948bc12a2ee9`;
- validation-panel contract fingerprint: `9ab65d47fa2db7dec2525e42f3fac2b05351e498b7dff454fec2b3dea9ce7a64`;
- memory mode: adaptive, with the declared 12-block d256 paper-exact configuration; and
- an atomic 50.7 MB resume checkpoint is present.

The evaluator checkpoints every 512 segments. At the current rate that is approximately every 18 minutes, so a Colab interruption should lose at most about one checkpoint interval of compute. Rerunning 03m with the same inputs is intended to resume rather than restart.

## Current quantitative snapshot

The most recent live status and the most recent durable statistics snapshot are slightly offset because live progress is written more frequently than the 512-segment checkpoint.

| Quantity | Observation | Interpretation |
|---|---:|---|
| Live segments | 18,816 / 225,220 (8.35%) | Evaluation incomplete |
| Live valid bases | 3,247,372 / 41,615,672 panel bases (7.80%) | Incomplete coverage |
| Completed/started streams | 7 / 8 | All are from the first accession |
| Completed accessions | 0 / 8 | No accession-level endpoint is final |
| Rate | 0.474 segments/s | About 131.9 h total projected for held-out evaluation |
| Durable snapshot | 18,432 segments, 3,173,644 bases | Used for partial metrics below |
| Partial BPB | 1.994386 | Descriptive only; not the gate statistic |
| Partial top-1 / top-2 accuracy | 0.247% / 0.452% | Above uniform guessing, still low in absolute terms |
| Gradient interventions | 0 | Exact recurrence stayed unmodified; this is not proof of benign gradients |
| Active fast state | finite in all 12 blocks | Necessary numerical condition passed at the checkpoint boundary |

The durable snapshot covers only `GCF_000351405.1` (ANI99 group `ani99_000014`), a scaffold assembly with ten streams. It had completed seven streams and was partway through the eighth. The 18,432 durable segments are 56.0% of that accession's 32,901 planned segments and only 8.18% of the full panel plan.

The partial 1.994386 BPB is lower than the earlier 512-segment C19 diagnostic (2.427433 BPB) and just below the 2-BPB independent-base reference. That is encouraging as an observation, but it does not demonstrate that memory improved predictions with context. The evaluated segments and streams changed as the run progressed, so context adaptation, local sequence difficulty, and contig composition are confounded.

It is also invalid to claim a c16 improvement from the available numbers. The c16 pilot evaluated only 256 segments per accession (2,048 total, 0.86% panel coverage), whereas the current C19 aggregate is a long sequential prefix of one accession. The samples and memory histories are not matched.

## What went well

1. **The correct final C19 checkpoint is being evaluated.** Its hash matches the archived final training identity.
2. **The run is reproducibly pinned.** Code, dataset, panel, model configuration, device, and work-plan hashes are stored in the resume contract.
3. **The uncapped plan is genuinely broad.** It declares all eight held-out ANI99-separated accessions and 225,220 segments rather than silently reusing the capped terminal diagnostic.
4. **Long-sequence execution remains finite so far.** The active fast weights, surprise tensors, and short histories in all 12 blocks were finite at the durable checkpoint.
5. **Exact resume infrastructure is functioning operationally.** A current atomic evaluation checkpoint and live cursor exist, which makes a multi-session evaluation recoverable.
6. **The partial loss is less alarming than the 512-segment diagnostic.** The one-accession running aggregate reached 1.994386 BPB, although this is not yet a fair gate result.

## What does not look good

### 1. The analysis is much slower than an interactive Colab job

The held-out evaluation alone projects to about 5.5 days of continuous A100 time. The current notebook is sequential and still must run memory behavior, memory tracing, generation, and the gate afterward. A single uninterrupted Colab session is unlikely to be the operational unit; repeated exact resumes must be treated as the normal workflow.

### 2. The baseline required by the gate does not exist

The configured `c17_v3_c16_broad_baseline_resumable` directory has a completed pilot only. It lacks both frozen inputs needed by `seqtrainer-titans-stage-c-scale-gate`:

- `full_evaluation/evaluation.json`; and
- `generation_t0p6/generation_evaluation.json`.

This is a hard workflow blocker, not a scientific failure of C19. Notebook 03j must be rerun on an A100 with `RUN_FULL_A100=True` to create them. Thresholds or paths should not be changed after observing C19.

### 3. Long-horizon memory diagnostics have escalated dramatically

Across the first 18,432 evaluation segments, the reported means were:

| Diagnostic | Partial held-out mean | Final 500 training-step mean | Approximate ratio |
|---|---:|---:|---:|
| Retrieval norm | 17,185.06 | 391.65 | 43.9x |
| Memory-update norm | 10,698.81 | 27.69 | 386x |
| Surprise norm | 641.44 | 1.244 | 516x |
| State-drift norm | 75,014.79 | 271.24 | 277x |

The maximum raw and conditioned memory-gradient RMS reached 1,123,338.625. C19 training's largest recorded raw memory-gradient RMS was 45.58, making the held-out maximum roughly 24,600 times larger. No value became nonfinite, and no intervention occurred because the paper-exact condition has no configured memory-gradient limiter. Thus `gradient_intervention_fraction = 0` means fidelity to the declared recurrence, not numerical comfort.

At the durable cursor, the serialized fast weights were still finite, with per-block L2 norms from 1.80 to 11.12 and absolute maxima from 0.279 to 4.679. This argues against a simple persistent weight explosion. The very large activation/gradient telemetry may instead be transient or input/state dependent. Either way, it is a serious long-horizon conditioning signal that must be localized by stream, coordinate, and block before an E100 continuation.

### 4. The current partial result has no panel-level inferential value

No held-out accession is complete. Seven of eight accessions have not started. A median across accessions, paired bootstrap, fraction improved, and full-corpus mean cannot be computed. The current BPB must not be compared directly with the frozen median threshold of 1.96.

### 5. The actual gate components are absent

There is no evidence yet for the preregistered requirements of positive immediate and delayed association in at least ten blocks, generation diversity preservation, >=10% six-mer JSD improvement, lower GC error, or improvement in three generation metric families.

## Scientific meaning

The partial run provides two useful scientific observations, but no confirmatory conclusion.

First, C19 can carry paper-exact adaptive memory through thousands of sequential held-out segments while preserving finite serialized state. This is stronger long-context operational evidence than the original 512-segment terminal diagnostic.

Second, the long held-out trajectory reveals a conditioning regime that short diagnostics and typical training windows did not expose: extreme memory gradients and very large retrieval, update, surprise, and drift telemetry can coexist with finite compact fast weights and an aggregate BPB near 2.0. This is scientifically relevant because it suggests that behavioral usefulness and numerical conditioning must be evaluated separately. Finite loss alone is not enough to characterize the memory mechanism as stable or useful.

What the run does **not** yet show is broad E. coli generalization, adaptive-memory benefit, improved key/value association, successful genomic generation, biological function, or a reason to scale exposure. Those claims require the completed paired and controlled endpoints.

## Impact on the E100 plan

**No-go for now; the plan remains at the E25 evaluation gate.**

The evidence does not justify abandoning C19: the training completed, the current evaluator is resumable, and the partial BPB is potentially promising. But it also does not justify E100. Three independent conditions block progression:

1. 03m is only 8.35% through its first stage;
2. the full c16 comparison artifacts are missing; and
3. the long-horizon memory-gradient behavior requires explicit review even if the frozen gate later passes.

Do not alter the frozen E25 thresholds. Do not interpret a partial BPB below 2.0 as a substitute for `proceed: true`. E100 should be a conditional continuation only after the complete gate passes and the terminal resume-verification issue documented in the C19 final review is resolved or formally supplemented.

## Recommended next steps

1. **Let the current 03m evaluation continue while the runtime is alive.** Do not delete or move `evaluation/resume/adaptive.evaluation.pt`.
2. **After any disconnect, rerun the same 03m cells unchanged.** Confirm that live status reports `resumed: true` and that the completed-segment count continues from the saved cursor.
3. **Complete the missing c16 baseline independently.** Run 03j on an A100 with `RUN_FULL_A100=True`; require both the uncapped `full_evaluation/evaluation.json` and matched `generation_t0p6/generation_evaluation.json`.
4. **Finish candidate evaluation before interpreting prediction performance.** Require all eight per-accession BPBs, full coverage, paired bootstrap versus c16, and the frozen median/improvement/fraction criteria.
5. **Rerun 03m after candidate evaluation completes.** Its guard will skip the completed evaluation and proceed to memory behavior, trace, generation, and gate compilation.
6. **Audit the held-out conditioning tail.** From the completed trace or an explicitly labeled diagnostic run, identify the accession, stream, segment, coordinate, and blocks responsible for the 1.12M memory-gradient RMS maximum. Preserve this as exploratory safety evidence; do not rewrite the frozen gate.
7. **Resolve terminal resume verification.** Demonstrate terminal scheduler exhaustion on the immutable final checkpoint and deterministic continuation from an immutable pre-exhaustion checkpoint.
8. **Make the decision mechanically.** Proceed to E100 only when `scale_analysis_v1/gate/scale_gate.json` exists and contains `"proceed": true`; otherwise stop and diagnose generalization, data exposure, or memory conditioning.

## Evidence boundary and limitations

This is a complete audit of the evidence currently produced by 03m, not a completed 03m outcome report. Live and resume files are mutable while the Colab process runs. Exact source identities at the review boundary are recorded in `SOURCE_MANIFEST.json`, and the extracted statistics are stored in `03m_interim_snapshot.json`.

