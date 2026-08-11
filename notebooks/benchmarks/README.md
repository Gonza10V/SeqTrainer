# Benchmark Notebooks

This is the legacy benchmark-notebook index. The current shared model results
are maintained under `notebooks/benchmarks_sg/` and hosted accelerator entry
points under `notebooks/colab_benchmarks/`.

All current model comparisons use the same predefined GSE144621 train,
validation, and test CSV files, seed `42`, validation-only MCC thresholding,
and held-out test reporting.

## CNN Benchmark

- [`cnn_benchmark/`](cnn_benchmark/): CNN reference and CNN-v2 promoter
  classification benchmarks on the predefined train/eval/test split.

The legacy CLI examples below cover the CNN benchmark path. The current shared
model harness and DNABERT2/iPro-MP records are documented in the benchmark
folders listed above:

```bash
seqtrainer benchmark run config-examples/benchmarks/cnn.toml
seqtrainer benchmark run config-examples/benchmarks/cnn_v2.toml
seqtrainer benchmark compare outputs/benchmarks/cnn* --output-dir outputs/benchmarks/comparison
```

Use `--base-dir` when running from outside the repository root, and
`--output-dir` to override the configured output folder:

```bash
seqtrainer benchmark run config-examples/benchmarks/cnn_v2.toml \
  --output-dir outputs/benchmarks/cnn_v2_trial
```

`cnn.toml` is the reference CNN configuration. `cnn_v2.toml` is the regularized
CNN-v2 candidate using AdamW, OneCycleLR, dropout, validation-MCC checkpoint
selection, and early stopping.

## Colab Notebook Links

- [CNN reference benchmark](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/benchmarks/cnn_benchmark/cnn_reference_benchmark_colab.ipynb)
- [CNN-v2 final benchmark](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/benchmarks/cnn_benchmark/cnn_v2_final_benchmark_colab.ipynb)

These links use the shared model-baselines branch. After it merges, replace the
branch name with `dev` or, preferably, pin a reviewed commit SHA.

## Shared Outputs

Completed CNN benchmark runs should write:

- `metrics.csv`
- `metrics.json`
- `predictions.csv`
- `manifest.json`
- `history.csv`
- `checkpoints/`

## Comparison Table Template

| Model | Config | Selection Metric | Test MCC | Test AUPRC | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| CNN reference | `cnn.toml` | validation MCC |  |  | exact CNN baseline |
| CNN-v2 | `cnn_v2.toml` | validation MCC |  |  | regularized CNN candidate |
| DNABERT2 final training | `notebooks/final_training/config/dnabert2_final_training_t4.toml` | validation MCC | 0.192182 | 0.624236 | canonical shared split; full fine-tuning |

## Reproducibility Rule

Benchmark notebooks should record:

- dataset source and split files
- model configuration
- random seed
- selected validation threshold
- validation metrics used for model choice
- held-out test metrics used for final reporting
