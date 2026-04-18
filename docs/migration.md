# Migration guide

This guide documents current migration paths from legacy/prototype APIs to the stabilized package structure.

## Deprecation timeline

SeqTrainer now emits standardized `DeprecationWarning`s for legacy compatibility paths.

- **Current status (April 18, 2026):** soft deprecation (warnings + compatibility still available)
- **Planned earliest removal date:** **December 31, 2026**
- **Target release for removals:** **0.4.0**

> Compatibility paths are retained to support migration, but new development should use the replacement APIs below.

## Old → New module mappings

| Legacy module / command | Replacement | Notes |
|---|---|---|
| `seqtrainer.preprocessing` | `seqtrainer.transforms.dna` | Legacy wrapper remains; emits deprecation warnings. |
| `seqtrainer.dataset_builder` | `seqtrainer.data.sbol` | Legacy wrapper remains; emits deprecation warnings. |
| `seqtrainer.gnn` | `seqtrainer.graph` + `seqtrainer.torch` | Prototype module deprecated; use graph extraction/config + torch model modules. |
| `seqtrainer build-dataset ...` | `seqtrainer dataset build ...` | Legacy CLI alias remains; emits deprecation warning. |

## New APIs to target

- `seqtrainer.clients.SynBioHubClient`
- `seqtrainer.sparql.*` (recipes/builders/typed normalization)
- `seqtrainer.data.*` (`DatasetRecipe`, `MaterializedDataset`, cache, tensorization)
- `seqtrainer.graph.*` (schema extraction + config builders)
- `seqtrainer.torch.*` / `seqtrainer.keras.*` adapters and model helpers

## CLI migration

Prefer the structured commands:

- `seqtrainer dataset recipes`
- `seqtrainer dataset build <files...> [--recipe ... --cache ... --output ...]`
- `seqtrainer model build --framework <torch|keras> ...`
- `seqtrainer sparql prefixes`
- `seqtrainer inspect-sbol <file>`

## Notes for maintainers

If extending compatibility shims, route all warning behavior through `seqtrainer._deprecation.warn_deprecated` so timelines and wording stay consistent.
