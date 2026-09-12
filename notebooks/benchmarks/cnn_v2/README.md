# CNN Promoter Benchmark

This folder contains the reproducible CNN-v2 benchmark for the shared *E. coli*
promoter-classification CSV splits. CNN-v2 is the CNN reference for DNABERT2
and iPro-MP comparisons: all models use the same predefined train,
validation, and held-out test files.

## Result

| Run | Test MCC | Test AUPRC | Status |
| --- | ---: | ---: | --- |
| CNN reference, 10 cycles | 0.187208 | 0.618783 | Reproduced baseline |
| CNN-v2, 50 cycles | **0.220884** | **0.645976** | Best CNN result |
| CNN-v2, 100 cycles | 0.208165 | 0.634116 | Did not improve |

The 50-cycle candidate was selected from validation MCC before its held-out
test metrics were reported. The 100-cycle run shows that additional cycles did
not improve this CNN configuration.

Detailed candidate metrics, model settings, and the comparison chart are in
[assets/RESULTS.md](assets/RESULTS.md). The cross-model summary remains in
[Results_Final.md](../../../Results_Final.md).

## Run

Open [the CNN-v2 Colab notebook](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-model-baselines-clean/notebooks/benchmarks/cnn_v2/cnn_v2_final_benchmark_colab.ipynb)
and run it top to bottom with a GPU runtime. It uses Drive CSVs when available,
then falls back to the bundled archive.

To run through the package from the repository root:

```bash
python -m pip install -e ".[torch]"
python -m zipfile -e data/data_DNABERT/promoter_classification_DNABERT.zip data/promoter_classification
seqtrainer benchmark run notebooks/benchmarks/cnn_v2/config.toml
```

For a short local smoke run, copy the TOML, set `training.max_epochs = 1`
and change `[outputs].output_dir`, then run it through the same command:

```bash
seqtrainer benchmark run path/to/cnn_v2_smoke.toml
```

## Dataset And Policy

Source accession: [GSE144621](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144621).
The three required split files are:

- `train_EP_DNA_BERT2_genomic_order.csv`
- `eval_EP_DNA_BERT2_genomic_order.csv`
- `test_EP_DNA_BERT2_genomic_order.csv`

The bundled archive is
`data/data_DNABERT/promoter_classification_DNABERT.zip`. A Drive copy is
available in the [AIxBio promoter classification folder](https://drive.google.com/drive/folders/1rH47oJEjQjkJvHXKX_rwDjDb--dGPGx2).

Thresholds are selected on validation MCC only. Test MCC and AUPRC are final
held-out reporting metrics, accompanied by accuracy, balanced accuracy,
precision, recall, F1, sensitivity, specificity, AUROC, and a confusion
matrix.

Each completed package benchmark writes `metrics.csv`, `metrics.json`,
`history.csv`, `manifest.json`, `predictions.csv`, and a checkpoint under
`checkpoints/`.
