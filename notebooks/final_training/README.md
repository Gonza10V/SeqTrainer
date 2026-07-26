# Final training notebooks

This folder contains the final Colab-facing model workflows. dnabert1.ipynb is a short filename for full DNABERT2-117M fine-tuning. ipromp1.ipynb is the official iPro-MP inference-only workflow.

## Colab links

- https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/final_training/dnabert1.ipynb
- https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/final_training/ipromp1.ipynb

Canonical templates, intentionally unchanged:
- notebooks/colab_benchmarks/dnabert2_finetune_t4_colab.ipynb
- notebooks/colab_benchmarks/ipromp_t4_colab.ipynb

## Run

Open a link, choose Runtime -> Change runtime type -> T4 GPU, then execute from the top. Both notebooks clone issue-3-all-model-baselines, install the working dependency chain, and stage data/model files on /content.

Use the same three CSV files as CNN:
- train_EP_DNA_BERT2_genomic_order.csv
- eval_EP_DNA_BERT2_genomic_order.csv
- test_EP_DNA_BERT2_genomic_order.csv

Keep RUN_MODE="quick" for the first run. smoke is only a setup check; final runs the selected full split. Drive is used for persistence and /content for active computation.

## Scientific contract

Both models use the exact shared train/validation/test files, sequence and label fields, seed 42, and binary labels 0/1. The notebooks reject invalid DNA and cross-split duplicate sequences. Thresholds are selected on validation only with MCC over the 0.05 to 0.95 grid in 0.01 increments; ties use validation AUPRC, validation balanced accuracy, then closeness to 0.5. Test metrics are read only after selection.

DNABERT2 first reproduces the successful full-fine-tuning T4 baseline, then performs a small resumable search over learning rate, epochs, effective batch size, weight decay, warmup and dropout. It keeps the existing SeqTrainer model/tokenizer/runner path. It does not switch to frozen DNABERT, original DNABERT, LoRA, or an unsupported scheduler.

iPro-MP preserves the official DNABERT/DNABERT-6 model, overlapping 6-mer conversion, E. coli species 10 model and five official folds. It is inference-only in this repository, so epochs and learning-rate optimization do not apply.

## Outputs

Drive paths:
- SeqTrainer/outputs/final_training/dnabert1/
- SeqTrainer/outputs/final_training/ipromp1/

Runs save baseline_metrics, sweep results, best_config, metrics, predictions, manifest, history when training occurs, environment, split_summary, checkpoints and plots. After both models finish, ipromp1 writes model_comparison.csv and model_comparison.md under SeqTrainer/outputs/final_training, ranked by test MCC then test AUPRC.

RESUME=True restores completed runs. DNABERT2 quick search is capped at 12 trials and has a 10.5 hour guard. A CUDA OOM is retried once with half physical batch and doubled accumulation. GPU type and Colab session duration are not guaranteed.

The notebooks were statically validated in the repository environment; no live Colab GPU execution or MCC improvement is claimed until you run them.
