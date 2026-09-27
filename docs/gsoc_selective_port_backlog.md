# Local GSoC Selective-Port Backlog

## Local epic

This document replaces an unavailable GitHub-issue workflow with a local, tracked
implementation backlog.  It audits the original fourteen GSoC tasks against the
repository state rather than treating a large historical branch as a merge plan.

**Audit baseline.** `main` is `a1bd573`.  `gsoc_shreeya` is `9788136`; it is
audit evidence only and must not be merged or wholesale cherry-picked.  The local
feature tips `feat/titans-paper-mac-stage-b` (`9894523`) and
`feat/stage-c-03q-context-eval` (`513b63c`) are also valid local completion
evidence.  Neither is evidence that work is on `main`, released, or compatible
with its current public interfaces.

Future work starts from current `main` on a narrowly named branch.  Select only
the paths needed for the task, reconcile them with current interfaces, and update
this record with the resulting main-based commit and executed checks.

## Integration receipt: selective foundation

- **Base:** `main@a1bd573931b69da08e0e5c205c7ac7262cf4e56f`; branch
  `codex/gsoc-selective-integration`; dirty-state fingerprint at final check:
  `66b6ebbb03c3bb8bbb6aee325cf4d9badbd0c3fab62aa9c44780ed5ca468ba76`.
- **Included:** Python/CI foundation; deterministic scalar KNN; pinned lazy
  hosted-model presets; benchmark config/split/manifest/comparison contract;
  CNN, DNABERT2, and external iPro-MP runners; reproducible bacterial manifest,
  sampling, split, and shard utilities; the first experimental Titans reference
  and deterministic CPU checkpoint/resume workflow.
- **Explicitly not included:** historical notebooks, reports, downloaded data,
  weights, generated artifacts, GPU evidence, and the unaccepted historical
  Stage-C source modules.
- **Executed (Python 3.12):** `mypy` (pass, 43 files); `ruff check src tests`
  (pass); `pytest -q` (pass, 158 tests); `git diff --check` (pass); and
  `uv build --out-dir /private/tmp/seqtrainer-selective-dist` (pass, sdist and
  wheel). `twine check` is **not run** because Twine is not installed in the
  available environment. Accelerator, remote-download, and external-executable
  checks are **not run** by design; no such absence is counted as a pass.

### Dependency map

`1 -> 2 -> (3, 4) -> (5, 6, 7) -> 8 -> 9`; independently, `10 -> 11 -> 13 ->
14`, with `12 -> 13`.  Task 10 is the canonical Titans boundary; tasks 11--14
must not introduce or preserve a second competing Titans stack.

### Global constraints and quality gates

- Preserve current `main` public APIs unless a task explicitly carries a
  deprecation/migration plan. Do not port unrelated removals or repository-wide
  rewrites from the audit branches.
- Pin remote model/checkpoint revisions; keep optional heavyweight dependencies
  optional; never check in datasets, model weights, or oversized generated
  benchmark artefacts.
- Benchmark results need explicit configuration, deterministic split provenance,
  validation-only selection, and a small manifest before comparison.
- Run, as applicable: `ruff check src tests`, `mypy src`, focused `pytest`, then
  the full suite; `python -m build`, `twine check dist/*`, and `git diff --check`.
  Compile notebooks and use isolated CPU/GPU smoke commands where a notebook or
  accelerator workflow is changed.
- A focused evidence sweep on `gsoc_shreeya` passed 28 tests:
  `.venv/bin/python -m pytest -q tests/test_knn_retrievers.py
  tests/test_benchmark_config.py tests/test_benchmark_harness.py
  tests/test_cnn_baseline.py tests/test_dnabert2_benchmark.py
  tests/test_bacteria_titan_dataset.py
  tests/test_titans_paper_mac_stage_b_infrastructure.py
  tests/test_titans_paper_mac_stage_c_smoke.py`.

**Status legend:** **complete** = locally implemented and evidenced outside
`main`; **partial** = useful evidence exists but an architectural, portability,
or acceptance gap remains; **not started** = no usable evidence; **superseded**
= deliberately replaced by a documented current-main alternative; **needs
follow-up** = implementation exists but requires a bounded verification or
decision.  “Complete” in this document never means merged.

## Ordered summary

| # | task | status | dependency | local evidence | next action |
|---:|---|---|---|---|---|
| 1 | Tutorial foundations | complete | — | `gsoc_shreeya@9788136` | Selectively port tutorials that still fit main. |
| 2 | Foundation-model presets | partial | 1 | `gsoc_shreeya@9788136` | Rebuild pinned presets against main. |
| 3 | Scalar KNN retrieval | complete | 1 | `gsoc_shreeya@9788136` | Port model and unit tests. |
| 4 | Shared benchmark core | complete | 3 | `gsoc_shreeya@9788136` | Port a minimal benchmark contract first. |
| 5 | CNN/CNN-v2 benchmarks | complete | 4 | `gsoc_shreeya@9788136` | Port runners/configs without result binaries. |
| 6 | iPro-MP adapter | complete | 4 | `gsoc_shreeya@9788136` | Port adapter behind an optional external boundary. |
| 7 | DNABERT2 | complete | 4 | `gsoc_shreeya@9788136` | Port pinned runner and offline tests. |
| 8 | Benchmark comparison | complete | 5–7 | `gsoc_shreeya@9788136` | Port manifest-only comparison. |
| 9 | Hosted benchmark notebooks | partial | 8 | `gsoc_shreeya@9788136` | Recreate notebooks pinned to a main commit. |
| 10 | Canonical Titans foundation | partial | — | `feat/titans-paper-mac-stage-b@9894523` | Choose one public Titans implementation. |
| 11 | Titans Stage B backends | partial | 10 | `feat/titans-paper-mac-stage-b@9894523` | Port only behind the canonical boundary. |
| 12 | Bacterial Titans data preparation | complete | — | `gsoc_shreeya@9788136` | Port reproducible metadata/shard tools. |
| 13 | Titans Stage C workflow | partial | 11, 12 | `feat/stage-c-03q-context-eval@513b63c` | Reduce to a main-compatible workflow. |
| 14 | Stage C reproducibility notebooks/evidence package | needs follow-up | 13 | `feat/stage-c-03q-context-eval@513b63c` | Curate small, reproducible evidence only. |

## 1. Tutorial foundations

**Goal and result.** Establish introductory, runnable package tutorials.  Local
evidence is the tutorial series added through `1bc7d43`, `36de4e1`, and
`b2a9e18`: `notebooks/tutorials/00_quickstart.ipynb` through the CNN examples.
They are absent from `main`, so this is locally complete, not integrated.

**Handoff.** Branch `feat/port-tutorial-foundations` from `main`. Keep each
notebook small, offline by default, and aligned with current imports; do not
port stale outputs, large data, or old package restructuring. Update the README
index and add notebook compilation/smoke coverage. Acceptance: notebook compile,
a clean-kernel smoke, relevant `pytest`, and the global gates.

## 2. Foundation-model presets

**Goal and result.** Provide promoter-task presets for Nucleotide Transformer v2,
HyenaDNA, Evo 2, and Gemma 3. Evidence includes
`src/seqtrainer/torch/backbones.py` plus tutorials 06--09 at `9788136`; commits
`81355f3`, `d5eeda3`, `d2e888d`, and `a8ff383` correct model choices. This is
partial because the branch changes surrounding torch interfaces and hosted-model
availability/revision pins must be revalidated on `main`.

**Handoff.** Branch `feat/port-foundation-presets` after task 1. Expose a small
documented preset registry, pin every revision, use lazy optional imports, and
test construction with mocked/offline loaders. Do not download weights in tests
or add compatibility wrappers for deleted branch APIs. Update dependency/docs;
run focused torch tests, notebook compile, then global gates.

## 3. Scalar KNN retrieval

**Goal and result.** Add scalar-label KNN retrieval and a promoter-activity
tutorial. `src/seqtrainer/models/knn.py`, exports, tutorial 10, and
`tests/test_knn_retrievers.py` were added in `90ae7d8`--`b54859b`; the focused
test sweep passed. Local completion evidence: `gsoc_shreeya@9788136`.

**Handoff.** Branch `feat/port-scalar-knn`. Preserve explicit fit/predict input
validation and deterministic neighbor/tie behavior; do not couple it to Titans
or benchmark machinery. Port source, exports, tests, and tutorial together;
accept with `pytest -q tests/test_knn_retrievers.py` plus lint/type/package gates.

## 4. Shared benchmark core

**Goal and result.** Establish common config, split, artifact, manifest, runner,
policy, and comparison contracts. `src/seqtrainer/benchmarks/` and
`tests/test_benchmark_{config,harness}.py` are present at `9788136`; the focused
tests passed. Later branch commits add split-content digests, leakage rejection,
and stale-artifact cleanup, which are important completion evidence.

**Handoff.** Branch `feat/port-benchmark-core` after task 3. Start with the
smallest stable TOML/config + manifest + split-provenance interface. Preserve
validation-only selection and reject cross-dataset/leaky comparisons. Do not
import all model runners or historical artifacts. Update config documentation;
accept with the two focused suites and a generated tiny fixture manifest.

## 5. CNN/CNN-v2 benchmarks

**Goal and result.** Provide configured CNN baselines and CNN-v2 benchmark
notebook support. Evidence: `src/seqtrainer/torch/cnn_baseline.py`,
`config-examples/benchmarks/{cnn,cnn_v2}.toml`, `tests/test_cnn_baseline.py`, and
the CNN-v2 notebook/config at `9788136`; focused tests passed.

**Handoff.** Branch `feat/port-cnn-benchmarks` after task 4. Port only runners
using the shared contract, record effective architecture and validation-only
candidate selection in manifests, and retain split/leakage checks. Exclude
generated SVG/result outputs. Add CPU tiny-data regression tests and a notebook
compile check.

## 6. iPro-MP adapter

**Goal and result.** Adapt iPro-MP inputs/outputs to the benchmark contract.
Evidence: `src/seqtrainer/adapters/ipromp*.py`, `config-examples/benchmarks/ipromp*.toml`,
and `notebooks/benchmarks/ipromp/` at `9788136`; commits `149aade` and
`837342d` cover device propagation, IDs, k-mer size, and validation.

**Handoff.** Branch `feat/port-ipromp-adapter` after task 4. Keep iPro-MP an
optional external executable/service boundary; preserve FASTA IDs and normalize
external predictions before scoring. Do not vendor weights or external project
code. Add fixture-based adapter tests and a skipped/explicit integration smoke;
update usage and failure documentation.

## 7. DNABERT2

**Goal and result.** Add frozen/fine-tuned DNABERT2 benchmark support. Evidence:
`src/seqtrainer/benchmarks/dnabert2.py`,
`src/seqtrainer/torch/dnabert2_benchmark.py`, three DNABERT2 configs, notebook,
and `tests/test_dnabert2_benchmark.py`; focused tests passed. Commits `f5edf2f`
and `9f44a79` document fallback-weight and primary-metric safeguards.

**Handoff.** Branch `feat/port-dnabert2-benchmark` after task 4. Pin model
revision/tokenizer, distinguish frozen from fine-tuned configuration, and test
offline with fakes. Do not make downloads implicit or select on test data.
Persist a manifest with revision, split digest, seeds, and metric; run focused
tests and a deliberately tiny opt-in smoke.

## 8. Benchmark comparison

**Goal and result.** Compare compatible benchmark artifacts. Evidence:
`src/seqtrainer/benchmarks/compare.py`, `artifacts.py`, and the final summary SVG
at `9788136`; `b490b53` requires manifests and `cb27c07` rejects cross-dataset
comparisons. The benchmark focused sweep passed.

**Handoff.** Branch `feat/port-benchmark-comparison` after tasks 5--7. Compare
only small manifests and result tables with matching dataset/split/config
identity; ignore explicitly skipped runs. Do not commit presentation figures or
invent a leaderboard. Add mismatch/rejection tests and document the generated
comparison command.

## 9. Hosted benchmark notebooks

**Goal and result.** Supply hosted/Colab/Kaggle benchmark entry notebooks.
The branch contains CNN-v2, iPro-MP, and DNABERT2 notebooks and repeatedly pins
them (`4ab1850`, `d22b2ca`). It is partial: their pinned revision is the audit
branch, not a selectively ported `main` commit, so they are not a deployable
main workflow.

**Handoff.** Branch `feat/port-hosted-benchmark-notebooks` after task 8. Generate
or rewrite each notebook against a fixed main SHA and an explicit runtime/dependency
cell; clear outputs and keep data/weights external. Verify JSON validity,
compilation, and one isolated smoke per host. Document cost, secrets, and expected
artifacts without committing host outputs.

## 10. Canonical Titans foundation

**Goal and result.** Introduce one canonical Titans/MAC foundation. The local
history contains `torch/titans.py`, `torch/titans_mac/`, and
`torch/titans_paper_mac/` with Stage A docs/tests (notably `677dab2` and
`46dd015`). This is partial: those overlapping stacks conflict with the
single-canonical-boundary requirement, despite substantial test and fidelity
evidence.

**Handoff.** Branch `feat/port-canonical-titans`. Decide and document one public
module/API, then port only that implementation with state/checkpoint semantics,
causal masking, and synthetic-stream tests. Do not preserve duplicate public
stacks or compatibility aliases merely to carry branch code forward. Update
architecture/migration docs; accept with canonical unit/parity tests and a CPU
smoke before any accelerator work.

## 11. Titans Stage B backends

**Goal and result.** Add measured backend variants (exact/approximate memory,
convolution, SDPA attention, and long-context studies). Evidence is
`torch/titans_paper_mac_stage_b/`, Stage B docs/artifacts, notebook, and tests at
`feat/titans-paper-mac-stage-b@9894523`. This is partial because it is built on
the non-canonical paper-MAC package and its generated artifacts are not suitable
for a narrow source port.

**Handoff.** Branch `feat/port-titans-stage-b` after task 10. Re-express backends
behind the selected canonical interface, retain explicit fidelity/telemetry and
portable math-SDPA fallback, and benchmark in generated local output directories.
Do not claim acceleration without parity plus measured evidence; do not commit
large matrices/SVGs. Run backend/parity tests, CPU smoke, optional GPU smoke, and
record commands/environment in a small manifest.

## 12. Bacterial Titans data preparation

**Goal and result.** Build reproducible bacterial source manifests, sampling,
splits, token shards, and Stage C streams. Evidence:
`src/seqtrainer/data/bacteria_titan/`, `scripts/build_bacteria_titan_dataset_colab.py`,
`docs/bacteria_titan.md`, and `tests/test_bacteria_titan_dataset.py` at
`9788136`; the focused test passed.

**Handoff.** Branch `feat/port-bacteria-titans-data`. Preserve source checksums,
accession/split provenance, deterministic sampling, and lazy shard access. Do
not commit downloaded genome data or bake notebook paths into package code. Add
tiny synthetic-source fixtures and a manifest validation command; update the
data documentation and run focused tests.

## 13. Titans Stage C workflow

**Goal and result.** Provide tokenizer, stream, training, checkpoint/resume,
capacity, evaluation, and generation workflow. The Stage C package, scripts,
handoffs, and tests are present on `gsoc_shreeya`; the local feature branch adds
later context/anomaly work through `513b63c`. This is partial because it depends
on the unresolved canonical Titans choice and brings a broad notebook/artifact
surface that cannot be selectively ported unchanged.

**Handoff.** Branch `feat/port-titans-stage-c` after tasks 11--12. Start with a
small CPU deterministic train/resume/evaluate slice using the canonical model
and manifest-backed bacterial data. Preserve checkpoint RNG/state restoration,
lazy streams, explicit device/SDPA contracts, and hardware preflight. Exclude
production reports, drive exports, and unbounded capacity claims. Run tokenizer,
model, workflow, and resume focused tests plus CPU/GPU smoke where available.

## 14. Stage C reproducibility notebooks and evidence package

**Goal and result.** Package reproducible Stage C notebook evidence and reports.
Evidence includes `notebooks/titans_stage_c/`, handoffs, study protocols, and the
C16 package on `9894523`; `513b63c` adds later C19 context-evaluation evidence.
It needs follow-up: the branch contains large PDFs, images, zip archives, and
host-specific outputs, so the evidence must be curated and rerun against a
main-based Stage C implementation before it is publishable as reproducible.

**Handoff.** Branch `docs/port-stage-c-evidence` after task 13. Keep source
notebooks, locked commands, small JSON/CSV manifests, checksums, and a concise
reproduction guide; regenerate figures into ignored output directories or an
external release store. Do not port binary reports, checkpoints, raw data, or
archive bundles. Validate notebook compilation, execute the documented isolated
smoke/reproduction path, verify manifests/checksums, and run `git diff --check`.

## Maintenance rule

When a task is selectively ported, replace its evidence entry with the
main-based branch/commit, retain the historical reference in prose if useful,
and record the exact commands and outcomes. Re-audit dependent rows rather than
assuming historical-branch compatibility.
