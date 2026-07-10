# SeqTrainer Benchmark Workspaces

This folder groups SeqTrainer project benchmark assets that are larger than the
basic tutorial notebooks.

## HPC/Alpine Workflows

Use [`sg_hpc/`](sg_hpc/README.md) for the Alpine/HPC workflows:

- `sg_hpc/dnabert2_alpine/`: DNABERT2 full fine-tuning on Alpine/A100-style hardware.
- `sg_hpc/ipromp_alpine/`: iPro-MP official E. coli pretrained five-fold ensemble inference.

These workflows are separated from Colab notebooks so the Slurm scripts, TOML
configs, setup scripts, and patch files can be uploaded and run on Alpine without
searching through notebook-specific folders.
