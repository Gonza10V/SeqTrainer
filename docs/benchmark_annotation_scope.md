# Benchmark and Annotation Scope

This document defines the scientific scope for reproducible bacterial promoter
prediction benchmarks and plasmid promoter annotation in SeqTrainer. It is meant
for researchers who want to understand what is being evaluated, what evidence is
needed to compare models, and what outputs should be reproducible.

## Summary

SeqTrainer will benchmark promoter prediction models on shared DNA sequence
examples, using shared splits, shared metrics, and recorded provenance. The
starting reference is the existing CNN tutorial baseline. Candidate models such
as DNABERT2 and iPro-MP should be adopted for annotation only when they improve
relevant benchmark metrics or provide a clearly justified scientific or
operational tradeoff.

The first annotation target is plasmid promoter prediction. The expected output
is a structured table of predicted promoter calls and an SBOL annotation draft
that preserves enough provenance for another researcher to inspect and rerun the
workflow.

## Existing SeqTrainer Capabilities

SeqTrainer already provides several building blocks for this work:

- SBOL-to-table extraction in `seqtrainer.data.sbol`.
- DNA preprocessing in `seqtrainer.transforms.dna`, including sequence
  normalization, padding/trimming, one-hot encoding, GC content, and k-mer
  features.
- Seeded train/validation/test splitting through
  `seqtrainer.data.materialized.MaterializedDataset`.
- A framework-neutral model registry stub in `seqtrainer.models.registry`.
- Optional PyTorch and Keras adapter namespaces.
- A command-line foundation with `inspect-sbol`, `build-dataset`, and SPARQL
  helper commands.
- Tutorial notebooks in `notebooks/tutorials/` for SBOL extraction, DNA
  features, dataset splits, CNN classification, and CNN regression.

The current CNN classification tutorial is a demonstration baseline. It uses 40
sample SBOL XML files from `data/sbol_data/`, derives binary labels by
thresholding the numeric `target` at the median, pads/trims sequences to a fixed
length, one-hot encodes the sequences, splits with seed `42`, and trains a small
PyTorch Conv1D model for 10 cycles with unweighted cross-entropy loss.

## Scientific Question

The benchmark asks:

> Given bacterial DNA sequence records or candidate sequence windows, how well
> can a model predict promoter-positive versus promoter-negative examples in a
> reproducible way?

Each benchmark example should contain:

- a DNA sequence or sequence window,
- a binary promoter label,
- source and provenance metadata,
- split assignment,
- stable identifiers that connect predictions back to the original sequence or
  SBOL source.

All model families must be evaluated on the same examples and split definitions.
The model selected for annotation should then be applied to plasmid candidate
windows to generate predicted promoter calls.

## Data and Labels

Labels may come from either:

1. curated binary promoter/non-promoter annotations, when available; or
2. a documented threshold over a numeric `target` value, when using the current
   tutorial-style SBOL data.

When numeric targets are thresholded, the threshold must be saved with the
results. The tutorial median threshold is acceptable for reproducing the
demonstration CNN baseline. However, benchmark results intended to support
scientific claims should explain why the chosen threshold is appropriate for the
dataset.

Every dataset row should preserve:

- source file or source record identifier,
- sequence identifier when available,
- numeric target when used,
- binary label,
- threshold or label rule,
- parsing or validation warnings where applicable.

## Initial Benchmark Scope

The initial benchmark should include:

- reproduction of the existing CNN classification baseline,
- a shared benchmark interface for CNN, DNABERT2, and iPro-MP,
- shared seeded train/validation/test splits,
- shared metric computation,
- JSON/CSV metrics output,
- prediction tables,
- run manifests with environment, seed, split, model version, threshold, and
  hyperparameters,
- class distribution reporting and an explicit imbalance-handling policy,
- DNABERT2 frozen-embedding baseline, with classifier-head fine-tuning only if
  the package path is stable,
- iPro-MP setup, smoke testing, wrapper or adapter evaluation, and conversion
  between SeqTrainer split tables and iPro-MP-compatible FASTA/prediction
  tables,
- unified model comparison on the same held-out test split,
- a model decision record explaining which model path should be used for
  plasmid annotation.

## Initial Annotation Scope

The initial annotation workflow should:

1. generate sliding windows or candidate promoter regions from plasmid
   sequences,
2. score those windows with the selected model,
3. merge overlapping positive windows into promoter feature calls,
4. export predicted promoters to a structured table,
5. export an SBOL annotation draft,
6. evaluate predictions against a curated held-out plasmid set when such a set
   is available.

The first version may use simple sliding windows and basic overlap merging.
Strand-aware post-processing, promoter-CDS association, confidence bands, and
genome-scale annotation are follow-on improvements unless they are required for
the curated plasmid evaluation.

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

## Required Metrics

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

For imbalanced datasets, AUPRC, MCC, balanced accuracy, sensitivity, and
specificity should be emphasized over plain accuracy. If a split contains only
one observed class, metrics that require both classes should be reported as
undefined or skipped with an explicit warning.

Candidate models should be compared against the reproduced CNN baseline.
DNABERT2, iPro-MP, or any improved model should be selected for annotation only
if it improves relevant metrics or provides a clearly justified tradeoff, such
as better reproducibility, lower operational complexity, or more useful
annotation behavior. Accuracy alone is not sufficient evidence of improvement.

## Reproducibility Requirements

Each benchmark run should write:

- metrics JSON,
- metrics CSV or an appendable summary table,
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

A researcher should be able to reproduce a reported result by installing
SeqTrainer, obtaining the documented data and model files, and running the
recorded configuration against the recorded split.

## Annotation Outputs

The plasmid annotation workflow should produce two primary artifacts.

### Promoter Call Table

The structured promoter table should include:

- plasmid or sequence ID,
- start and end coordinates,
- strand when supported,
- model score or probability,
- threshold,
- predicted label,
- supporting window IDs,
- run or model identifier.

### SBOL Annotation Draft

The SBOL annotation draft should contain predicted promoter features and
provenance sufficient to distinguish model predictions from existing curated
features.

## Reproduction Target

Another researcher should be able to:

1. build the same dataset or load the same materialized dataset snapshot,
2. inspect the label rule and class distribution,
3. reuse the same train/validation/test split,
4. rerun the CNN baseline,
5. rerun the DNABERT2 baseline path,
6. rerun the iPro-MP baseline path or documented iPro-MP adapter path,
7. compare models with the same metric code,
8. inspect the model decision record,
9. run the selected model on plasmid candidate windows,
10. regenerate promoter call tables and SBOL annotation drafts.

## Scope Checklist

This scope is satisfied when it:

- identifies the current CNN reference baseline,
- defines the promoter prediction and plasmid annotation tasks,
- states label and source assumptions,
- separates initial scope from follow-on work,
- lists required benchmark metrics,
- defines reproducibility outputs,
- defines annotation artifacts,
- gives future implementation work a clear scientific target.
