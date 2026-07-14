# Preparing SeqTrainer Offline Bundles For Alpine

Build this bundle on an internet-enabled workstation, then transfer it to Alpine. Slurm jobs should only execute already-staged code, data, containers, and model files.

## Bundle Contents

```text
seqtrainer-offline/
  image/seqtrainer-alpine-gpu.sif
  repository/SeqTrainer/
  models/DNABERT-2-117M/
  models/DNABERT-6/
  models/ipromp_ecoli/
  data/promoter_classification/
  manifests/
```

## Preparation Checklist

1. Build `containers/seqtrainer_alpine_gpu.def` into `image/seqtrainer-alpine-gpu.sif`.
2. Copy this SeqTrainer checkout into `repository/SeqTrainer`.
3. Stage the DNABERT2 snapshot in `models/DNABERT-2-117M`.
4. Stage DNABERT-6 files in `models/DNABERT-6`.
5. Stage iPro-MP E. coli fold checkpoints in `models/ipromp_ecoli`.
6. Copy the three shared promoter CSVs into `data/promoter_classification`.
7. Write `manifests/repository_revision.txt` with the SeqTrainer commit SHA.
8. Write `manifests/offline_bundle_manifest.json` with model revisions and data sources.
9. Generate `manifests/SHA256SUMS`.

You can create the directory skeleton and checksums with:

```bash
python tools/offline_prep/prepare_offline_bundle.py \
  --bundle-root /path/to/seqtrainer-offline \
  --repo-dir /path/to/SeqTrainer
```

Then validate before transfer:

```bash
python tools/offline_prep/validate_offline_bundle.py \
  --bundle-root /path/to/seqtrainer-offline \
  --model dnabert2

python tools/offline_prep/validate_offline_bundle.py \
  --bundle-root /path/to/seqtrainer-offline \
  --model ipromp
```

Validation failure exits with code `2` and lists missing or mismatched files.
