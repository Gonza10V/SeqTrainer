# SeqTrainer agent instructions

## Colab and fresh-environment dependency rules

- Treat every Colab runtime as a fresh Python environment. Do not assume that packages present in a developer virtualenv or a previous Colab session are available.
- SeqTrainer's top-level import eagerly reaches `seqtrainer.data.sbol`. Any environment that imports `seqtrainer` must provide the direct runtime dependencies declared in `pyproject.toml`, including `rdflib>=6.3.2`, `requests>=2.31`, and `sbol2>=1.4`.
- An editable install using `pip install --no-deps -e ...` is allowed only after the notebook or script explicitly installs every required direct dependency and task-specific extra. Never use `--no-deps` as a substitute for dependency resolution.
- A module import such as `import seqtrainer.torch...` still executes `seqtrainer/__init__.py`; it does not bypass the SBOL/RDF dependency chain.
- Every changed Colab bootstrap must run a smoke check with the exact interpreter used for the job. At minimum, import `rdflib`, `requests`, `sbol2`, and `seqtrainer`, plus the job's CLI module. Print captured stderr on failure.
- Do not use pytest collection inside a production Colab notebook as an environment gate. Run repository tests during development; use focused imports and the job's own preflight/runtime projection in Colab.
- When practical, validate bootstrap changes in an empty virtualenv or container. Passing tests in the repository's existing virtualenv is not sufficient evidence that a fresh Colab bootstrap is complete.

## Generated Stage C notebooks

- Edit the notebook builder under `scripts/`, regenerate the `.ipynb`, and test that every generated code cell compiles. Do not hand-edit only the generated notebook.
- Keep immutable model and dataset reads on Colab-local storage after a single staging pass. Keep high-frequency evaluator writes off Google Drive.
- For 03q, mount Drive once and persist validated current/previous resume ZIP generations at stage boundaries and after each one-hour model session. The verified ANI inputs are `inputs/ecoli_skani_triangle.tsv` and `stage_c_dataset/manifests/ani99_membership.parquet` under `SeqTrainerStageC`.
- Before reporting a Drive-side diagnostic, check whether execution reached the first persistence point. An absent resume ZIP means there will be no Drive log to inspect.
