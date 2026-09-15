#!/usr/bin/env python3
"""Render the frozen figures for the first completed 03m accession."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "accession1_results.json").read_text())
COLORS = {"c19": "#2A6F97", "c16": "#E09F3E", "warning": "#9E2A2B", "neutral": "#6C757D"}


def save_bpb() -> None:
    comparison = DATA["descriptive_comparisons"]
    labels = ["C19 terminal\n512-segment prefix", "c16 pilot\n256-segment prefix", "C19 complete\naccession"]
    values = [
        comparison["c19_terminal_512_segment_prefix"]["bits_per_base"],
        comparison["c16_256_segment_pilot_prefix"]["bits_per_base"],
        DATA["c19_complete_accession"]["bits_per_base"],
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    bars = ax.bar(labels, values, color=["#8AB6D6", COLORS["c16"], COLORS["c19"]], width=0.62)
    ax.axhline(2.0, color=COLORS["neutral"], linestyle="--", linewidth=1.5, label="Independent-base reference (2.00)")
    ax.axhline(1.96, color=COLORS["warning"], linestyle=":", linewidth=2, label="E25 median gate reference (1.96)")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.4f}", ha="center", fontweight="bold")
    ax.set_ylim(1.85, 2.52)
    ax.set_ylabel("Bits per base (lower is better)")
    ax.set_title("GCF_000351405.1: complete C19 result and descriptive prefixes")
    ax.legend(loc="upper right", frameon=False)
    ax.text(0.01, -0.20, "Caution: prefix/model comparisons have different coverage and memory histories; they are not the frozen paired gate.", transform=ax.transAxes, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(ROOT / "accession1_bpb_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_coverage() -> None:
    comparison = DATA["descriptive_comparisons"]
    labels = ["c16 pilot prefix", "C19 terminal prefix", "C19 complete accession"]
    segments = [
        comparison["c16_256_segment_pilot_prefix"]["segments"],
        comparison["c19_terminal_512_segment_prefix"]["segments"],
        DATA["c19_complete_accession"]["segments"],
    ]
    bases = [
        comparison["c16_256_segment_pilot_prefix"]["valid_bases"],
        comparison["c19_terminal_512_segment_prefix"]["valid_bases"],
        DATA["c19_complete_accession"]["valid_bases"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3))
    for ax, values, title, ylabel in zip(axes, (segments, bases), ("Evaluated segments", "Evaluated predictable bases"), ("Segments (log scale)", "Bases (log scale)")):
        bars = ax.bar(labels, values, color=[COLORS["c16"], "#8AB6D6", COLORS["c19"]])
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=18)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:,}", ha="center", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Why the available BPB comparisons are not coverage-matched", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ROOT / "accession1_coverage_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_streams() -> None:
    items = list(DATA["stream_segments"].items())
    labels = [key.replace("NZ_", "") for key, _ in items]
    values = np.array([value for _, value in items])
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(9.4, 6.1))
    bars = ax.barh(np.array(labels)[order], values[order], color=COLORS["c19"])
    for bar, value in zip(bars, values[order]):
        ax.text(value + max(values) * 0.012, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", fontsize=9)
    ax.set_xlabel("Segments evaluated")
    ax.set_title("Complete coverage of all 10 replicon/scaffold streams")
    ax.set_xlim(0, max(values) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "accession1_stream_coverage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    save_bpb()
    save_coverage()
    save_streams()

