"""Build a descriptive C16/C19 generation comparison from two JSON reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


SHARED_GROUPS = ("temperature_0.8", "temperature_1")


def read_report(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} is not a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group(report: Mapping[str, object], name: str) -> Mapping[str, float]:
    summary = report["distribution_summary"]
    if not isinstance(summary, Mapping) or not isinstance(summary.get(name), Mapping):
        raise ValueError(f"missing distribution group {name!r}")
    return summary[name]  # type: ignore[return-value]


def prodigal_group(report: Mapping[str, object], name: str) -> Mapping[str, float]:
    prodigal = report["prodigal"]
    if not isinstance(prodigal, Mapping) or prodigal.get("status") != "completed":
        raise ValueError("Prodigal results are unavailable")
    groups = prodigal.get("groups")
    if not isinstance(groups, Mapping) or not isinstance(groups.get(name), Mapping):
        raise ValueError(f"missing Prodigal group {name!r}")
    return groups[name]  # type: ignore[return-value]


def scalar_metrics(report: Mapping[str, object], name: str) -> dict[str, float]:
    reference = group(report, "reference")
    generated = group(report, name)
    kmer = report["kmer_jsd_to_heldout_reference"]
    if not isinstance(kmer, Mapping) or not isinstance(kmer.get(name), Mapping):
        raise ValueError(f"missing k-mer JSD group {name!r}")
    return {
        "six_mer_jsd": float(kmer[name]["6"]),  # type: ignore[index]
        "overlapping_six_mer_diversity_ratio": float(generated["unique_6mer_fraction"])
        / max(float(reference["unique_6mer_fraction"]), 1e-12),
        "gc_absolute_error": abs(
            float(generated["gc_fraction"]) - float(reference["gc_fraction"])
        ),
        "entropy_absolute_error": abs(
            float(generated["base_entropy_bits"])
            - float(reference["base_entropy_bits"])
        ),
        "homopolymer_absolute_error": abs(
            float(generated["max_homopolymer"])
            - float(reference["max_homopolymer"])
        ),
    }


def prodigal_metrics(report: Mapping[str, object], name: str) -> dict[str, float]:
    reference = prodigal_group(report, "reference")
    generated = prodigal_group(report, name)
    return {
        "coding_density_absolute_error": abs(
            float(generated["coding_density"]) - float(reference["coding_density"])
        ),
        "genes_per_10kb_absolute_error": abs(
            float(generated["genes_per_10kb"])
            - float(reference["genes_per_10kb"])
        ),
        "median_gene_bases_absolute_error": abs(
            float(generated["median_gene_bases"])
            - float(reference["median_gene_bases"])
        ),
        "median_intergenic_bases_absolute_error": abs(
            float(generated["median_intergenic_bases"])
            - float(reference["median_intergenic_bases"])
        ),
    }


def plot_metrics(
    comparison: Mapping[str, Mapping[str, Mapping[str, float]]],
    output: Path,
    *,
    prodigal: bool,
) -> None:
    keys = (
        (
            "coding_density_absolute_error",
            "Coding-density absolute error",
            "absolute fraction error",
        ),
        ("genes_per_10kb_absolute_error", "Gene-density absolute error", "genes / 10 kb"),
        ("median_gene_bases_absolute_error", "Median gene-length error", "bases"),
        (
            "median_intergenic_bases_absolute_error",
            "Median intergenic-length error",
            "bases",
        ),
    ) if prodigal else (
        ("six_mer_jsd", "6-mer distribution divergence", "JSD bits"),
        (
            "overlapping_six_mer_diversity_ratio",
            "Overlapping 6-mer diversity ratio",
            "generated / held-out",
        ),
        ("gc_absolute_error", "GC-fraction absolute error", "absolute fraction error"),
        ("homopolymer_absolute_error", "Maximum-homopolymer error", "bases"),
    )
    colors = {"c16": "#7A5AA6", "c19": "#168F91"}
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.5), constrained_layout=True)
    title = "Prodigal calibration" if prodigal else "Sequence-distribution calibration"
    fig.suptitle(
        f"C16 versus C19 {title.lower()}\nDescriptive only: generation protocols are not matched",
        fontsize=16,
        weight="bold",
    )
    temperatures = ("0.8", "1.0")
    x = np.arange(len(temperatures))
    width = 0.34
    for axis, (key, panel_title, ylabel) in zip(axes.flat, keys, strict=True):
        for offset, model in ((-width / 2, "c16"), (width / 2, "c19")):
            values = [
                comparison[model][f"temperature_{temperature:g}"][key]
                for temperature in map(float, temperatures)
            ]
            bars = axis.bar(x + offset, values, width, label=model.upper(), color=colors[model])
            axis.bar_label(bars, fmt="%.3g", padding=2, fontsize=8)
        if key == "overlapping_six_mer_diversity_ratio":
            axis.axhline(1.0, color="#D85E52", linestyle="--", linewidth=1, label="reference")
        axis.set_title(panel_title)
        qualifier = "1 is ideal" if "ratio" in key else "lower is better"
        axis.set_ylabel(f"{ylabel}\n({qualifier})", fontsize=9)
        axis.set_xticks(x, temperatures)
        axis.set_xlabel("temperature", fontsize=9)
        axis.grid(axis="y", color="#D9E2EA", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c16", type=Path, required=True)
    parser.add_argument("--c19", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    c16 = read_report(args.c16)
    c19 = read_report(args.c19)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    identity_keys = ("dataset_fingerprint", "taxonomy_manifest_sha256", "split", "species")
    identity = {
        key: {"c16": c16.get(key), "c19": c19.get(key), "matched": c16.get(key) == c19.get(key)}
        for key in identity_keys
    }
    protocol_keys = ("prompt_tokens", "new_tokens", "temperatures", "top_k", "top_p", "seed")
    protocols = {key: {"c16": c16.get(key), "c19": c19.get(key)} for key in protocol_keys}
    comparison = {
        model: {
            name: scalar_metrics(report, name)
            for name in SHARED_GROUPS
        }
        for model, report in (("c16", c16), ("c19", c19))
    }
    prodigal = {
        model: {
            name: prodigal_metrics(report, name)
            for name in SHARED_GROUPS
        }
        for model, report in (("c16", c16), ("c19", c19))
    }
    output = {
        "format_version": 1,
        "classification": "descriptive_unmatched_protocol_comparison",
        "source_sha256": {"c16": sha256(args.c16), "c19": sha256(args.c19)},
        "identity": identity,
        "protocols": protocols,
        "shared_temperature_metrics": comparison,
        "shared_temperature_prodigal_metrics": prodigal,
        "limits": [
            "Prompt length, continuation length, seed, and temperature grids are not matched.",
            "JSD and diversity estimates depend on sampled sequence length.",
            "Only four prompt accessions were sampled in each run.",
            "This comparison cannot determine the frozen E25 generation gate.",
        ],
    }
    (args.output_dir / "descriptive_comparison.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_metrics(comparison, args.output_dir / "c16_c19_sequence_calibration.png", prodigal=False)
    plot_metrics(prodigal, args.output_dir / "c16_c19_prodigal_calibration.png", prodigal=True)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
