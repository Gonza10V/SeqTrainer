# Migration guide

## Module moves

- `seqtrainer.preprocessing` -> `seqtrainer.transforms.dna`
- `seqtrainer.dataset_builder` -> `seqtrainer.data.sbol`
- `seqtrainer.gnn` prototype -> `seqtrainer.graph` + `seqtrainer.torch`

Compatibility wrappers are retained for `preprocessing` and `dataset_builder`.

## New APIs

- `seqtrainer.clients.SynBioHubClient`
- `seqtrainer.data.DatasetRecipe`
- `seqtrainer.data.MaterializedDataset`
- `seqtrainer.applications.build_promoter_regression_blueprint`

## CLI

Use:

- `seqtrainer sparql prefixes`
- `seqtrainer inspect-sbol <file>`
- `seqtrainer build-dataset <files...>`


## CLI migration

- New structured command group: `seqtrainer dataset ...`
  - `seqtrainer dataset recipes`
  - `seqtrainer dataset build <files...>`
- Legacy `seqtrainer build-dataset` remains as a compatibility alias for now.
- New framework-specific config command: `seqtrainer model build --framework <torch|keras>`.
