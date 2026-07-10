# SeqTrainer SG HPC Workflows

This folder collects the Alpine/HPC workflows for the SeqTrainer promoter
benchmark. It is intentionally separate from the Colab notebooks so the HPC
submission scripts, TOML configs, and runtime patches are easy to find.

## Folder Layout

| Folder | Purpose | Main entry point |
| --- | --- | --- |
| `dnabert2_alpine/` | DNABERT2 full fine-tuning on Alpine/A100-style hardware | `run_dnabert2_finetune_alpine.sbatch` |
| `ipromp_alpine/` | Official iPro-MP E. coli pretrained five-fold ensemble inference | `run_ipromp_alpine.sbatch` |

## Shared Scientific Contract

Both workflows are designed to stay comparable with the CNN-v2 benchmark:

- same promoter-classification task;
- same predefined train/validation/test split filenames;
- seed `42`;
- no threshold tuning on the test set;
- threshold selected on validation MCC only;
- final ranking by held-out test MCC first and test AUPRC second;
- artifacts written as CSV/JSON manifests wherever possible.

The expected split files are:

```text
train_EP_DNA_BERT2_genomic_order.csv
eval_EP_DNA_BERT2_genomic_order.csv
test_EP_DNA_BERT2_genomic_order.csv
```

Use the AIxBio `Promoter Classification/Data` folder or the repository's
`data/data_DNABERT/promoter_classification_DNABERT.zip` source. Do not reshuffle
or regenerate splits for these model comparisons.

## DNABERT2 Alpine

Use `dnabert2_alpine/` for claim-bearing DNABERT2 fine-tuning when Colab is too
slow or too memory-constrained. The config keeps the same comparison surface as
CNN-v2, but uses the DNABERT2 backbone with the HPC resource profile documented
in that folder's README.

## iPro-MP Alpine

Use `ipromp_alpine/` for the official iPro-MP E. coli pretrained model 10
five-fold ensemble. The workflow downloads only `10_fold_1.pth` through
`10_fold_5.pth` from Zenodo record `15180139` and downloads DNABERT-6 from
Hugging Face model `zhihan1996/DNA_bert_6`.

The five iPro-MP folds are pretrained model folds, not SeqTrainer data folds.
Each fold scores the same validation/test records, the five probabilities are
averaged per sequence, validation MCC selects one threshold, and that threshold
is applied unchanged to test.

## Manual Values To Edit

Before submitting on Alpine, users usually only need to edit:

- Slurm allocation/account name in the `sbatch --account=...` command or script;
- optional partition/QoS if their allocation requires a different one;
- paths to copied data if not using the repository ZIP;
- wall time/memory if a site policy requires smaller values.

Do not edit seed, split filenames, threshold policy, or metric list when the goal
is direct comparison with CNN-v2, DNABERT2, and iPro-MP recorded results.
