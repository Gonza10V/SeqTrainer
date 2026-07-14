# DNABERT2 Alpine Offline Fine-Tuning

**OFFLINE ALPINE REQUIREMENT:** this job assumes the SeqTrainer repo, Apptainer image, data, and DNABERT2 snapshot already exist in `BUNDLE_ROOT`. The Slurm job must not clone code, install packages, or download model files.

## Benchmark Settings

| Setting | Value |
| --- | --- |
| Config | `config-examples/benchmarks/dnabert2_finetune_alpine_offline.toml` |
| Model | DNABERT2, local `/models/DNABERT-2-117M` |
| Mode | full encoder fine-tuning |
| Seed | `42` |
| Sequence max length | `128` tokenizer tokens |
| Batch size | `4` |
| Gradient accumulation | `8` |
| Effective batch size | `32` |
| Epochs | `5` |
| Learning rate | `3e-5` |
| Optimizer | AdamW |
| Precision | bf16 |
| Threshold | validation MCC only |
| Primary metrics | MCC, AUPRC |

## Run

```bash
cd /projects/$USER/seqtrainer-offline/repository/SeqTrainer
export BUNDLE_ROOT=/projects/$USER/seqtrainer-offline
sbatch notebooks/benchmarks_sg/sg_hpc/dnabert2_alpine/run_dnabert2_finetune_alpine.sbatch
```

Monitor:

```bash
tail -f seqtrainer-dnabert2-ft-<JOBID>.out
```

## What The Job Does

1. Validates the offline bundle with `tools/offline_prep/validate_offline_bundle.py`.
2. Runs `seqtrainer benchmark run` inside Apptainer with read-only binds for `/models` and `/data`.
3. Loads DNABERT2 from `/models/DNABERT-2-117M` only.
4. Saves comparable benchmark artifacts under `/work/outputs`.
5. Copies results to `/projects/$USER/seqtrainer-results/dnabert2/<JOBID>/`.

## Expected Local Model Files

`/models/DNABERT-2-117M` must contain:

- `config.json`
- `configuration_bert.py`
- `bert_layers.py`
- `bert_padding.py`
- `flash_attn_triton.py`
- `tokenizer.json` or `vocab.txt`
- `pytorch_model.bin` or `model.safetensors`

If any file is missing, the job exits before training.
