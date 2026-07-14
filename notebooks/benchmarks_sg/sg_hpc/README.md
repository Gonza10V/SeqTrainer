# SeqTrainer Alpine Offline Benchmarks

**OFFLINE ALPINE REQUIREMENT:** Alpine jobs in this folder are execution-only. Do not run `git`, `pip`, `conda`, Hugging Face downloads, Zenodo downloads, or source patching inside Slurm. Build and transfer the offline bundle before submitting jobs.

## What This Folder Runs

These workflows run the same promoter benchmark split used by CNN and Colab runs:

- train: `train_EP_DNA_BERT2_genomic_order.csv`
- validation: `eval_EP_DNA_BERT2_genomic_order.csv`
- test: `test_EP_DNA_BERT2_genomic_order.csv`
- labels: `1 = promoter`, `0 = non-promoter/background`
- seed: `42`
- threshold: selected on validation only, using MCC
- final reporting: held-out test metrics, primarily MCC and AUPRC

## Offline Bundle Layout

Stage this bundle outside Alpine, then transfer it to a project path such as `/projects/$USER/seqtrainer-offline`:

```text
seqtrainer-offline/
  image/seqtrainer-alpine-gpu.sif
  repository/SeqTrainer/
  models/
    DNABERT-2-117M/
    DNABERT-6/
    ipromp_ecoli/
      10_fold_1.pth
      10_fold_2.pth
      10_fold_3.pth
      10_fold_4.pth
      10_fold_5.pth
  data/promoter_classification/
    train_EP_DNA_BERT2_genomic_order.csv
    eval_EP_DNA_BERT2_genomic_order.csv
    test_EP_DNA_BERT2_genomic_order.csv
  manifests/
    offline_bundle_manifest.json
    repository_revision.txt
    SHA256SUMS
```

The Apptainer definition is at `containers/seqtrainer_alpine_gpu.def`. Build it on an internet-enabled machine, not inside the Slurm job.

## Validate Before Submitting

From Alpine login node:

```bash
export BUNDLE_ROOT=/projects/$USER/seqtrainer-offline
apptainer exec --nv --cleanenv \
  --bind "$BUNDLE_ROOT:/bundle:ro" \
  --bind "$BUNDLE_ROOT/repository/SeqTrainer:/repo:ro" \
  --bind "$BUNDLE_ROOT/models:/models:ro" \
  --bind "$BUNDLE_ROOT/data:/data:ro" \
  "$BUNDLE_ROOT/image/seqtrainer-alpine-gpu.sif" \
  python /repo/tools/offline_prep/validate_offline_bundle.py --bundle-root /bundle --model dnabert2
```

Use `--model ipromp` for the iPro-MP bundle check.

## Submit Jobs

DNABERT2 full fine-tuning:

```bash
cd /projects/$USER/seqtrainer-offline/repository/SeqTrainer
export BUNDLE_ROOT=/projects/$USER/seqtrainer-offline
sbatch notebooks/benchmarks_sg/sg_hpc/dnabert2_alpine/run_dnabert2_finetune_alpine.sbatch
```

iPro-MP E. coli five-fold external evaluation:

```bash
cd /projects/$USER/seqtrainer-offline/repository/SeqTrainer
export BUNDLE_ROOT=/projects/$USER/seqtrainer-offline
sbatch notebooks/benchmarks_sg/sg_hpc/ipromp_alpine/run_ipromp_alpine.sbatch
```

After `sbatch`, monitor with the printed job id:

```bash
squeue -u $USER
tail -f seqtrainer-dnabert2-ft-<JOBID>.out
tail -f seqtrainer-ipromp-<JOBID>.out
```

Replace `<JOBID>` with the number printed by `sbatch`.

## Outputs

Each completed run writes:

- `metrics.csv`
- `metrics.json`
- `predictions.csv`
- `manifest.json`
- `history.csv` when training occurs
- `checkpoints/` when training occurs

Default final copy locations:

- DNABERT2: `/projects/$USER/seqtrainer-results/dnabert2/<JOBID>/`
- iPro-MP: `/projects/$USER/seqtrainer-results/ipromp/<JOBID>/`

## Troubleshooting

- Missing model files: rebuild or restage the offline bundle; do not download inside Slurm.
- Invalid split CSVs: ensure every split has `sequence,label` and binary labels `0/1`.
- Permission errors: set `FINAL_OUTPUT_DIR` to a writable project directory before `sbatch`.
- GPU allocation errors: keep the configured `aa100` partition and A100 GPU requests unless Alpine support tells you otherwise.
