# iPro-MP Alpine Offline Evaluation

**OFFLINE ALPINE REQUIREMENT:** this job uses local DNABERT-6 files and local iPro-MP E. coli fold checkpoints only. It does not call the official inference script, clone repos, install packages, or download weights.

## Benchmark Settings

| Setting | Value |
| --- | --- |
| Config | `config-examples/benchmarks/ipromp_external_alpine_offline.toml` |
| Model | iPro-MP E. coli species `10` |
| Mode | external pretrained inference, no training |
| Folds | `10_fold_1.pth` through `10_fold_5.pth` |
| Ensemble | mean positive-class probability |
| DNABERT dependency | local `/models/DNABERT-6` |
| Tokenization | overlapping 6-mers |
| Token max length | `128` |
| Batch size | `16` |
| Epochs | not applicable; pretrained inference only |
| Threshold | validation MCC only |
| Primary metrics | MCC, AUPRC |

## Run

```bash
cd /projects/$USER/seqtrainer-offline/repository/SeqTrainer
export BUNDLE_ROOT=/projects/$USER/seqtrainer-offline
sbatch notebooks/benchmarks_sg/sg_hpc/ipromp_alpine/run_ipromp_alpine.sbatch
```

Monitor:

```bash
tail -f seqtrainer-ipromp-<JOBID>.out
```

## What The Job Does

1. Validates the offline bundle.
2. Converts the shared CSV splits to FASTA with `seqtrainer benchmark prepare-ipromp`.
3. Runs `python -m seqtrainer.adapters.ipromp_offline` on train, validation, and test FASTA files.
4. Scores the prediction CSVs with the shared benchmark evaluator.
5. Copies outputs and external prediction files to `/projects/$USER/seqtrainer-results/ipromp/<JOBID>/`.

## Expected Local Model Files

`/models/DNABERT-6` must contain:

- `config.json`
- `pytorch_model.bin`
- `vocab.txt`

`/models/ipromp_ecoli` must contain:

- `10_fold_1.pth`
- `10_fold_2.pth`
- `10_fold_3.pth`
- `10_fold_4.pth`
- `10_fold_5.pth`

The manifest records species id, fold names, ensemble method, token max length, and truncation statistics for each split.
