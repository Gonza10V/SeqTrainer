# SeqTrainer architecture

## Layering

1. **Domain core (framework-neutral)**
   - `seqtrainer.clients`
   - `seqtrainer.sparql`
   - `seqtrainer.data`
   - `seqtrainer.transforms`
   - `seqtrainer.models`

2. **Framework adapters (optional deps)**
   - `seqtrainer.keras`
   - `seqtrainer.torch`

3. **Graph utilities**
   - `seqtrainer.graph`

4. **Task applications**
   - `seqtrainer.applications`

5. **Delivery surfaces**
   - `seqtrainer.cli`

## Extension strategy

- Keep SBOL/SynBioHub semantics in domain modules, not in framework wrappers.
- Add new tasks as application blueprints that reference:
  - a dataset recipe,
  - transforms,
  - backbone/head choice,
  - adapter path.
- Use lazy imports in framework modules so base install remains lightweight.

## Experimental code policy

Prototype scripts and notebooks remain available, but stable APIs should be surfaced through modules under `seqtrainer/*`.


## Data snapshots and cache

`seqtrainer.data.cache` provides local snapshot persistence (`examples.jsonl` + versioned `manifest.json`) for reproducible dataset materialization workflows.


## Graph stabilization status

`seqtrainer.graph` now exposes tested schema extraction (`extract_node_types`, `extract_typed_edges`) and config builders (`GraphConfigSpec`, `build_config`, `write_config`) so graph workflows no longer depend on ad-hoc prototype scripts.
