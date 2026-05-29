# Benchmark and Annotation Scope

This document defines the scientific scope for reproducible promoter prediction
benchmarks and plasmid promoter annotation in SeqTrainer. It is written for
researchers who want to understand what is being evaluated, how results should
be compared, and what artifacts should be produced for reuse.

## Current Capabilities

SeqTrainer is currently a synthetic biology ML toolkit centered on SBOL and
SynBioHub data workflows. The package already has:

- SBOL helpers in `seqtrainer.data.sbol` that extract a sequence, numeric target,
  and source path from SBOL XML files.
- DNA transforms in `seqtrainer.transforms.dna` for normalization, fixed-length
  padding/trimming, one-hot encoding, GC content, and k-mer features.
- `MaterializedDataset` in `seqtrainer.data.materialized` for seeded
  train/validation/test splits.
- A framework-neutral model registry stub in `seqtrainer.models.registry`.
- Optional framework adapter namespaces for PyTorch and Keras.
- A CLI foundation with `inspect-sbol`, `build-dataset`, and SPARQL prefix
  helpers.
- Tutorial notebooks in `notebooks/tutorials/`, including:
  - `01_sbol_to_dataset.ipynb` for SBOL-to-table extraction.
  - `02_dna_features.ipynb` for DNA feature transforms.
  - `03_dataset_splits.ipynb` for reproducible splits.
  - `04_end_to_end_cnn_classification.ipynb` for a mini CNN classifier.
  - `05_end_to_end_cnn_regression.ipynb` for a mini CNN regressor.

The current CNN classification tutorial is a small demonstration baseline. It
uses the first 40 `data/sbol_data/sample_design_*.xml` files, derives a binary
label by thresholding the numeric `target` at the median, pads/trims sequences
to a fixed length, one-hot encodes them, splits with seed `42`, and trains a
small PyTorch Conv1D model for 10 cycles with unweighted cross-entropy loss.

## Scientific Task

The P0 benchmark task is binary bacterial DNA promoter prediction from sequence
windows or sequence records. Each example should contain:

- a DNA sequence or generated sequence window,
- a binary promoter label,
- source/provenance metadata,
- split assignment,
- enough identifiers to connect predictions back to the original sequence or
  SBOL source.

The benchmark should compare model families on the same examples, using the
same split definitions and metric code. The annotation workflow should then
apply the selected model path to plasmid candidate windows and export predicted
promoter calls.

## Label and Source Assumptions

P0 labels may come from two sources:

1. Curated binary promoter/non-promoter annotations when available.
2. A documented threshold over a numeric `target` value when using the current
   tutorial-style SBOL data.

When numeric targets are thresholded, the threshold must be recorded with the
results. The tutorial median threshold is acceptable for reproducing the
demonstration CNN baseline, but benchmark runs intended to support scientific
claims should state why the threshold is appropriate for the dataset being
evaluated.

Every dataset row should preserve at least:

- source file or source record identifier,
- sequence identifier when available,
- numeric target when used,
- binary label,
- threshold or label rule,
- parsing/validation warnings where applicable.

## Initial Benchmark Scope

The initial benchmark should include:

- Reproducible CNN baseline reproduction from the existing tutorial behavior.
- A shared benchmark interface for CNN, DNABERT2, and iPro-MP evaluation.
- Shared seeded train/validation/test splits.
- Shared metric computation and JSON/CSV output.
- Run manifests with environment, seed, split, model version, threshold, and
  hyperparameters.
- Imbalance handling policy that records class distributions and avoids
  test-set leakage.
- DNABERT2 frozen-embedding baseline, with optional classifier-head fine-tuning
  if the package path is stable.
- iPro-MP setup, wrapper or adapter path, smoke test, and conversion between
  SeqTrainer splits and iPro-MP-compatible FASTA/prediction tables.
- Unified comparison of CNN, DNABERT2, and iPro-MP on the same held-out split.
- A model decision record selecting the first annotation model path.
- Plasmid annotation workflow:
  - generate sliding windows or candidate regions,
  - score windows with the selected model,
  - merge overlapping positive windows,
  - export promoter calls as a structured table,
  - export an SBOL annotation draft,
  - evaluate against a curated held-out plasmid set when available.
- Reproducibility notebooks and a report explaining how to reproduce and extend
  the work.

## Initial Non-Goals

The initial benchmark and plasmid annotation workflow do not require:

- full genome-level E. coli annotation,
- retraining iPro-MP from scratch,
- cross-species iPro-MP transfer experiments,
- production-grade model serving,
- exhaustive hyperparameter search,
- replacing PyTorch, Keras, Hugging Face, or iPro-MP with SeqTrainer-native model
  implementations,
- treating tutorial-only median-threshold results as a final biological claim.

These items can be treated as follow-on work once the benchmark and plasmid
annotation workflow are working.

## Required Benchmark Metrics

Every benchmarked model path should report:

- accuracy,
- balanced accuracy,
- AUROC,
- AUPRC,
- F1,
- MCC,
- sensitivity/recall for the positive class,
- specificity for the negative class,
- confusion matrix.

For imbalanced datasets, AUPRC, MCC, balanced accuracy, and sensitivity/specificity
should be treated as more informative than plain accuracy. If a split has only
one observed class, metrics that require both classes must be reported as
undefined or skipped with an explicit warning rather than failing silently.

Candidate model paths should be compared against the reproduced CNN baseline.
DNABERT2, iPro-MP, or any improved model should only be selected for annotation
if the benchmark shows improved relevant metrics or a clearly justified tradeoff,
such as better reproducibility, lower operational complexity, or more useful
annotation behavior. Accuracy alone is not sufficient evidence of improvement.

## Reproducibility Requirements

Each benchmark run should write:

- metrics JSON,
- metrics CSV or appendable summary table,
- prediction table with example IDs and scores,
- split metadata,
- run manifest,
- configuration file or resolved configuration snapshot.

The run manifest should include:

- git commit and branch,
- package version or local editable install marker,
- Python version and relevant package versions,
- model family and model/checkpoint identifier,
- seed,
- split identifier,
- label rule and threshold,
- sequence preprocessing parameters,
- class distribution per split,
- hyperparameters,
- output directory.

Another researcher should be able to reproduce a reported result by installing
SeqTrainer, obtaining the documented data/model files, and running the recorded
configuration against the recorded split.

## Annotation Output Expectations

The plasmid annotation workflow should produce two primary artifacts:

1. A structured promoter call table containing:
   - plasmid or sequence ID,
   - start and end coordinates,
   - strand when supported,
   - model score or probability,
   - threshold,
   - predicted label,
   - supporting window IDs,
   - run/model identifier.

2. An SBOL annotation draft containing predicted promoter features and provenance
   sufficient to distinguish model predictions from existing curated features.

The first version may use a simple sliding-window candidate generator and basic
overlap merging. Strand-aware post-processing, promoter-CDS association, richer
confidence bands, and genome-level output are follow-on work unless they become
necessary for the curated plasmid evaluation.

## Researcher Reproduction Target

Another researcher should be able to:

1. build the same dataset or load the same materialized dataset snapshot,
2. inspect the label rule and class distribution,
3. reuse the same train/validation/test split,
4. rerun the CNN baseline,
5. rerun the DNABERT2 baseline path,
6. rerun the iPro-MP baseline path or documented iPro-MP adapter path,
7. compare the models with the same metric code,
8. inspect the model decision record,
9. run the selected model on plasmid candidate windows,
10. regenerate promoter call tables and SBOL annotation drafts.

## Document Success Criteria

This scope is sufficient when it:

- identifies the current code and notebook baseline,
- defines the primary prediction and annotation tasks,
- states label/source assumptions,
- separates P0 scope from P1/P2 follow-ons,
- lists required benchmark metrics,
- defines reproducibility outputs,
- defines plasmid annotation MVP artifacts,
- gives future implementation work a clear scientific target.
