# SeqTrainer

SeqTrainer is a synthetic biology ML domain toolkit focused on connecting **SBOL / SynBioHub data** to modern model workflows.

It is designed to be complementary to Keras and PyTorch rather than replacing them.

## What this refactor introduces

- Clear package layering under `seqtrainer/`
- Framework-neutral core (`clients`, `sparql`, `data`, `transforms`, `models`)
- Optional framework adapters (`seqtrainer.keras`, `seqtrainer.torch`)
- Graph-focused utilities in `seqtrainer.graph`
- Application-level API entrypoints in `seqtrainer.applications`
- CLI foundation (`seqtrainer` command)

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e '.[torch]'
pip install -e '.[keras]'
pip install -e '.[gnn]'
pip install -e '.[dev]'
```

## Package layout

- `seqtrainer/clients`: SynBioHub and remote clients (auth/retry/pagination-ready client)
- `seqtrainer/sparql`: prefixes, builders, canonical query recipes, and typed result normalization
- `seqtrainer/data`: SBOL loaders, recipes, materialized datasets, local snapshots/cache manifests
- `seqtrainer/transforms`: DNA transforms and feature extraction
- `seqtrainer/models`: framework-neutral backbone/head registries and default model specs
- `seqtrainer/keras`: Keras model factories + tensorized `tf.data` adapters from `MaterializedDataset` (optional dependency)
- `seqtrainer/torch`: PyTorch tensorized adapters/dataloaders, HF/DNABERT backbones, and regression/classification heads (optional dependency)
- `seqtrainer/graph`: RDF/SBOL graph conversion, schema extraction, and graph-config builders
- `seqtrainer/applications`: task-oriented blueprints
- `seqtrainer/cli`: command-line entrypoints

## CLI examples

```bash
seqtrainer sparql prefixes
seqtrainer inspect-sbol data/sbol_data/sample_design_0.xml
seqtrainer dataset recipes
seqtrainer dataset build data/sbol_data/sample_design_0.xml --cache
seqtrainer model build --framework torch --task regression
```

## Status

This is the first architecture-focused cleanup. Some framework integrations are intentionally placeholders with TODOs to keep a stable, minimal public surface.


## CI checks

The repository includes a GitHub Actions CI matrix for:

- base test suite
- `torch` extra
- `keras` extra
- `gnn` extra
- lint (`ruff`) and type checks (`mypy`)
- packaging build + `twine check`
