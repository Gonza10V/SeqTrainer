# Colab Benchmarks

This folder contains reproducible Google Colab entry points for SeqTrainer
benchmarks that need hosted accelerators.

## DNABERT2 Full Fine-Tuning

[`dnabert2_finetune_a100_colab.ipynb`](dnabert2_finetune_a100_colab.ipynb)
provides a larger Colab DNABERT2 full-fine-tuning profile.

The notebook keeps the scientific contract fixed:

- the same predefined train, validation, and test CSV files as CNN-v2;
- seed `42`;
- full DNABERT2 encoder fine-tuning;
- AdamW at `3e-5` with `0.1` warmup and `0.01` weight decay;
- physical batch size `4`, gradient accumulation `8`, effective batch size `32`;
- four maximum epochs with validation-MCC early stopping;
- validation-only threshold selection and final held-out test reporting;
- the same metrics and benchmark artifact format as the shared benchmark run.

The notebook requires an A100 because the canonical configuration uses BF16.
Dataset files are read from Google Drive, while model downloads and caches use
Colab-local storage. Checkpoints and final artifacts are written back to Drive.

The notebook pins the SeqTrainer implementation commit used by the benchmark
workflow, keeping the Colab executions tied to the same tested model
runner and configuration.

## DNABERT2 T4 Profile

[`dnabert2_finetune_t4_colab.ipynb`](dnabert2_finetune_t4_colab.ipynb)
runs the same pinned DNABERT2-117M backbone, shared split, seed, optimizer,
learning rate, pooling, validation-only threshold policy, metrics, and artifact
schema on a 16 GB T4.

The T4-specific changes are explicit in
[`config/dnabert2_finetune_t4.toml`](config/dnabert2_finetune_t4.toml):

- FP16 instead of A100 BF16;
- physical batch size `2` with gradient accumulation `16`, preserving effective
  batch size `32`;
- activation checkpointing to reduce memory;
- two maximum epochs instead of four, with validation-MCC early stopping.

This is a resource-constrained candidate, not a numerically identical replay of
the A100 run. Use it to obtain a real full-dataset T4 result, then confirm a
promising configuration with the longer full fine-tuning profile.

## Final Training DNABERT2 Profile

The final-training notebook and configuration are kept separately from the
short hosted-runtime profiles:

- [`../final_training/dnabert2-finetune-kaggle.ipynb`](../final_training/dnabert2-finetune-kaggle.ipynb)
- [`../final_training/config/dnabert2_final_training_t4.toml`](../final_training/config/dnabert2_final_training_t4.toml)

This run used the canonical shared split: 136,484 train rows, 19,498
validation rows, and 38,996 test rows. It used full encoder fine-tuning, seed
`42`, AdamW, learning rate `1e-5`, six maximum epochs, patience `2`, physical
batch size `2`, gradient accumulation `16` (effective batch size `32`), mean
pooling, token limit `104`, dropout `0.2`, FP16, and validation-MCC checkpoint
selection.

The recorded result was test MCC `0.192182` and test AUPRC `0.624236`. CNN-v2
50 cycles remains higher at test MCC `0.220884` and test AUPRC `0.645976`.
The archive was executed in Kaggle even though the TOML describes the intended
Colab T4 environment, so the runtime versions and exact SeqTrainer commit
must be pinned before claiming bit-for-bit reproduction.

## iPro-MP T4 Inference

[`ipromp_t4_colab.ipynb`](ipromp_t4_colab.ipynb) evaluates the same official
five-fold E. coli iPro-MP ensemble used by SeqTrainer. iPro-MP is
inference-only here, so reducing epochs does not apply. T4 memory and runtime
are controlled by:

- loading one official fold checkpoint at a time;
- inference batch size `4`;
- running validation and test by default;
- saving each completed split to Drive so a disconnected session can resume.

Train-split inference is optional because it is not used for threshold selection
or final held-out comparison. The five folds, 6-mer tokenization, sequence
coverage, validation-MCC threshold, test metrics, and output schema are not
reduced. Model weights require roughly 1.8 GB for the five iPro-MP checkpoints,
plus the DNABERT-6 backbone.

Latest recorded T4 inference result: validation-selected threshold `0.327886`,
held-out test MCC `0.068364`, test AUPRC `0.372180`, recall `0.234484`, and
specificity `0.823207`. The five official E. coli fold checkpoints are loaded
one at a time, each produces probabilities for the same validation/test rows,
and the final score is the arithmetic mean of those five probabilities.

## iPro-MP A100 Inference

[`ipromp_a100_colab.ipynb`](ipromp_a100_colab.ipynb) provides a larger Colab
iPro-MP external-inference workflow. It keeps the same official
E. coli model 10, five-fold ensemble, 6-mer tokenization, shared split, seed
`42`, validation-MCC threshold, and held-out test metrics.

iPro-MP is not trained in this benchmark, so epochs remain `0`. The A100 profile
only increases the inference batch size from `4` to `16`, matching the larger
profile and reducing wall-clock time without changing the model or scientific
comparison policy.

## Comparison Rule

The Colab notebooks keep the scientific comparison surface fixed: identical input
CSVs and labels, seed `42`, no test-set tuning, validation-selected threshold,
and held-out test MCC/AUPRC. Resource settings such as precision, physical batch
size, checkpointing, and epoch budget must be reported alongside the metrics.

## ProkBERT-mini T4 Profile

[![Open ProkBERT-mini T4 notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/colab_benchmarks/prokbert_mini_t4_colab.ipynb)

[`prokbert_mini_t4_colab.ipynb`](prokbert_mini_t4_colab.ipynb) adds the generic
`neuralbioinfo/prokbert-mini` microbial-DNA masked-language-model backbone to
the shared SeqTrainer promoter benchmark. ProkBERT is biologically relevant
because it was pretrained on broad microbial genomic sequence rather than on a
single promoter-classification task. The notebook deliberately does not use
`neuralbioinfo/prokbert-mini-promoter`; that promoter checkpoint could leak
task-related information. External ProkBERT promoter scores are therefore not
SeqTrainer benchmark results and are not directly comparable.

Place these unchanged files under `MyDrive/SeqTrainer/data/promoter_classification/`:

- `train_EP_DNA_BERT2_genomic_order.csv`
- `eval_EP_DNA_BERT2_genomic_order.csv`
- `test_EP_DNA_BERT2_genomic_order.csv`

Each CSV must contain `sequence` and `label`, with 300 bp sequences and binary
labels `1` and `0`. The notebook writes its fast local run to
`/content/seqtrainer_prokbert_run/`, then copies the complete artifact set to
`MyDrive/SeqTrainer/outputs/benchmarks/prokbert_mini_t4_full/` (or to the
separate `outputs/smoke_tests/prokbert_mini_t4_smoke_test/` directory in smoke
mode).

The T4 profile requests physical batch size `16`, gradient accumulation `2`
(effective batch size `32`), FP16, AdamW, learning rate `2e-5`, weight decay
`0.01`, warmup ratio `0.10`, three maximum epochs, gradient clipping `1.0`, and
validation-MCC checkpoint selection with patience `1`. If a genuine CUDA OOM is
observed in preflight, it explicitly falls back to physical batch size `8` and
accumulation `4`, preserving the effective batch size. The manifest records the
requested and effective settings and all pinned revisions.

Set `RUN_MODE = "smoke"` to sample each class in each split and run at most one
epoch. Smoke results are installation/tokenization/artifact diagnostics only;
they must never be entered into `Results_Final.md`. Set `RUN_MODE = "full"` to
use every row in the three predefined files. Resuming is enabled by default:
the notebook checks the config hash, split hashes, model revision, and
SeqTrainer commit before restoring model, optimizer, scheduler, scaler, history,
and RNG state.

The Hugging Face model is licensed `CC-BY-NC-4.0`; review that non-commercial
license before using the benchmark or resulting checkpoint in a commercial
setting. Only a completed full-data run on the canonical shared splits may be
reported in the final benchmark table.

