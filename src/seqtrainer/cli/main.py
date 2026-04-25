"""Command line interface for common SeqTrainer workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seqtrainer._deprecation import warn_deprecated
from seqtrainer.applications.promoter_regression import build_promoter_regression_blueprint
from seqtrainer.data import list_builtin_dataset_recipes, materialize_dataset_from_sbol
from seqtrainer.data.sbol import get_sequence_from_sbol
from seqtrainer.keras.factories import create_keras_model
from seqtrainer.sparql.prefixes import format_prefixes
from seqtrainer.torch.finetune import build_finetune_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seqtrainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_sbol = subparsers.add_parser("inspect-sbol", help="Inspect one SBOL file")
    inspect_sbol.add_argument("file", type=Path)

    dataset = subparsers.add_parser("dataset", help="Dataset recipe and materialization commands")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)

    recipes_cmd = dataset_sub.add_parser("recipes", help="List built-in dataset recipes")
    recipes_cmd.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    build_dataset = dataset_sub.add_parser("build", help="Build dataset from local SBOL files")
    build_dataset.add_argument("files", nargs="+", type=Path)
    build_dataset.add_argument("--recipe", default="local-sbol-regression")
    build_dataset.add_argument(
        "--y-uri",
        default="http://www.ontology-of-units-of-measure.org/resource/om-2/hasNumericalValue",
    )
    build_dataset.add_argument("--output", type=Path, help="Write dataset output to file")
    build_dataset.add_argument("--output-format", choices=["csv", "jsonl"], default="csv")
    build_dataset.add_argument("--cache", action="store_true", help="Write dataset snapshot cache manifest")
    build_dataset.add_argument("--cache-dir", type=Path)
    build_dataset.add_argument("--dataset-name", default="sbol-local")
    build_dataset.add_argument("--dataset-version")

    legacy_build = subparsers.add_parser("build-dataset", help="(Deprecated) alias for dataset build")
    legacy_build.add_argument("files", nargs="+", type=Path)
    legacy_build.add_argument("--recipe", default="local-sbol-regression")
    legacy_build.add_argument("--y-uri", default="http://www.ontology-of-units-of-measure.org/resource/om-2/hasNumericalValue")
    legacy_build.add_argument("--output", type=Path)
    legacy_build.add_argument("--output-format", choices=["csv", "jsonl"], default="csv")
    legacy_build.add_argument("--cache", action="store_true")
    legacy_build.add_argument("--cache-dir", type=Path)
    legacy_build.add_argument("--dataset-name", default="sbol-local")
    legacy_build.add_argument("--dataset-version")

    sparql = subparsers.add_parser("sparql", help="SPARQL helpers")
    sparql_sub = sparql.add_subparsers(dest="sparql_command", required=True)
    sparql_sub.add_parser("prefixes", help="Print default prefixes")

    model = subparsers.add_parser("model", help="Framework-specific model build commands")
    model_sub = model.add_subparsers(dest="model_command", required=True)

    model_build = model_sub.add_parser("build", help="Build framework model/fine-tune config")
    model_build.add_argument("--framework", choices=["torch", "keras"], default="torch")
    model_build.add_argument("--task", choices=["regression", "classification"], default="regression")
    model_build.add_argument("--backbone", default="dnabert2")
    model_build.add_argument("--head", default="regression-mlp")
    model_build.add_argument("--learning-rate", type=float, default=1e-4)
    model_build.add_argument("--epochs", type=int, default=5)
    model_build.add_argument("--output", type=Path)

    return parser


def _write_dataset_output(dataset_rows: list[dict], *, output_path: Path | None, output_format: str) -> None:
    if output_format == "csv":
        import pandas as pd

        payload = pd.DataFrame(dataset_rows).to_csv(index=False)
    elif output_format == "jsonl":
        payload = "\n".join(json.dumps(row, sort_keys=True) for row in dataset_rows)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported output format: {output_format}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + ("\n" if not payload.endswith("\n") else ""), encoding="utf-8")
    else:
        print(payload)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect-sbol":
        seq = get_sequence_from_sbol(args.file)
        print(f"sequence_length={len(seq) if seq else 0}")
        return 0

    if args.command == "dataset" and args.dataset_command == "recipes":
        recipes = list_builtin_dataset_recipes()
        if args.json:
            print(json.dumps({k: v.as_manifest_payload() for k, v in recipes.items()}, indent=2, sort_keys=True))
            return 0
        for name, recipe in recipes.items():
            print(f"{name}: sequence_field={recipe.sequence_field} label_field={recipe.label_field}")
        return 0

    if (args.command == "dataset" and args.dataset_command == "build") or args.command == "build-dataset":
        if args.command == "build-dataset":
            warn_deprecated(old="seqtrainer build-dataset", new="seqtrainer dataset build", kind="command", stacklevel=2)

        dataset, manifest = materialize_dataset_from_sbol(
            args.files,
            y_uri=args.y_uri,
            recipe_name=args.recipe,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            cache_dir=args.cache_dir,
            write_cache=args.cache,
        )
        _write_dataset_output(dataset.examples, output_path=args.output, output_format=args.output_format)
        if manifest:
            print(f"# cache_snapshot={manifest.snapshot_key}")
        return 0

    if args.command == "sparql" and args.sparql_command == "prefixes":
        print(format_prefixes())
        return 0

    if args.command == "model" and args.model_command == "build":
        if args.framework == "torch":
            payload = {
                "framework": "torch",
                "model_blueprint": build_promoter_regression_blueprint("torch"),
                "finetune": build_finetune_config(
                    backbone=args.backbone,
                    head=args.head,
                    task=args.task,
                    learning_rate=args.learning_rate,
                    epochs=args.epochs,
                ),
            }
        else:
            payload = {
                "framework": "keras",
                "model_blueprint": build_promoter_regression_blueprint("keras"),
                "factory_output": create_keras_model(
                    backbone=args.backbone,
                    head=args.head,
                    task=args.task,
                    learning_rate=args.learning_rate,
                    epochs=args.epochs,
                ),
            }

        text = json.dumps(payload, indent=2, sort_keys=True, default=str)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0

    parser.error("Unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
