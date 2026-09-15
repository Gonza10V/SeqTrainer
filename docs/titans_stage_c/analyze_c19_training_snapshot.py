"""Plot and summarize a downloaded c19 Stage C training snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROLLING_STEPS = 250
LONG_ROLLING_STEPS = 1000
WARMUP_BASES = 2_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rolling(frame: pd.DataFrame, column: str, window: int = ROLLING_STEPS) -> pd.Series:
    return frame[column].rolling(window, min_periods=max(20, window // 5)).mean()


def style_axis(axis: plt.Axes) -> None:
    axis.grid(color="#D9E2EA", linewidth=0.65, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)


def mark_warmup(axis: plt.Axes) -> None:
    axis.axvline(WARMUP_BASES / 1e6, color="#7B8794", linestyle="--", linewidth=1)


def load_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = (
        frame.sort_values("optimizer_step")
        .drop_duplicates("optimizer_step", keep="last")
        .reset_index(drop=True)
    )
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("training history contains non-finite numeric values")
    frame["cumulative_bases"] = frame["valid_bases"].cumsum()
    frame["million_bases"] = frame["cumulative_bases"] / 1e6
    return frame


def plot_learning(frame: pd.DataFrame, output: Path) -> None:
    colors = {
        "blue": "#2563A6",
        "cyan": "#168F91",
        "coral": "#D85E52",
        "gold": "#D99A2B",
        "navy": "#13233F",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), constrained_layout=True)
    fig.suptitle("c19 Medium adaptive E25: learning and optimization dynamics", fontsize=16, weight="bold")
    x = frame["million_bases"]

    axis = axes[0, 0]
    axis.scatter(x[::10], frame["bits_per_base"][::10], s=5, alpha=0.13, color=colors["blue"], label="per step")
    axis.plot(x, rolling(frame, "bits_per_base"), color=colors["blue"], linewidth=1.8, label="250-step mean")
    axis.plot(x, rolling(frame, "bits_per_base", LONG_ROLLING_STEPS), color=colors["navy"], linewidth=2.1, label="1,000-step mean")
    axis.axhline(2.0, color=colors["coral"], linestyle=":", linewidth=1.3, label="uniform-base reference")
    mark_warmup(axis)
    axis.set(title="a  Training bits per base", xlabel="processed bases (millions)", ylabel="BPB (lower is better)")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    style_axis(axis)

    axis = axes[0, 1]
    axis.plot(x, 100 * rolling(frame, "token_accuracy", 500), color=colors["blue"], linewidth=1.8, label="top-1")
    axis.plot(x, 100 * rolling(frame, "top_2_accuracy", 500), color=colors["cyan"], linewidth=1.8, label="top-2")
    mark_warmup(axis)
    axis.set(title="b  Exact next-6-mer accuracy", xlabel="processed bases (millions)", ylabel="rolling accuracy (%)")
    axis.legend(frameon=False)
    style_axis(axis)

    axis = axes[1, 0]
    axis.plot(x, frame["learning_rate"] * 1e5, color=colors["coral"], linewidth=1.7, label="learning rate ×10⁵")
    speed_axis = axis.twinx()
    speed_axis.plot(x, rolling(frame, "bases_per_second"), color=colors["cyan"], linewidth=1.5, label="bases/s")
    mark_warmup(axis)
    axis.set(title="c  Schedule and throughput", xlabel="processed bases (millions)", ylabel="learning rate (×10⁻⁵)")
    speed_axis.set_ylabel("bases/s")
    lines = axis.lines[:1] + speed_axis.lines
    axis.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8)
    style_axis(axis)
    speed_axis.spines["top"].set_visible(False)

    axis = axes[1, 1]
    axis.plot(x, rolling(frame, "gradient_norm"), color=colors["blue"], linewidth=1.8, label="outer gradient norm")
    memory_axis = axis.twinx()
    memory_axis.plot(x, rolling(frame, "raw_memory_gradient_rms_max"), color=colors["gold"], linewidth=1.5, label="max memory-gradient RMS")
    mark_warmup(axis)
    axis.set(title="d  Gradient dynamics", xlabel="processed bases (millions)", ylabel="outer norm before clipping")
    memory_axis.set_ylabel("memory-gradient RMS")
    lines = axis.lines[:1] + memory_axis.lines
    axis.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8)
    style_axis(axis)
    memory_axis.spines["top"].set_visible(False)

    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_memory(frame: pd.DataFrame, output: Path) -> None:
    colors = {
        "blue": "#2563A6",
        "cyan": "#168F91",
        "coral": "#D85E52",
        "gold": "#D99A2B",
        "purple": "#7A5AA6",
        "navy": "#13233F",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), constrained_layout=True)
    fig.suptitle("c19 Medium adaptive E25: neural-memory dynamics", fontsize=16, weight="bold")
    x = frame["million_bases"]

    axis = axes[0, 0]
    axis.plot(x, rolling(frame, "retrieval_norm"), color=colors["blue"], linewidth=1.8, label="retrieval norm")
    axis.plot(x, rolling(frame, "state_drift_norm"), color=colors["purple"], linewidth=1.8, label="state-drift norm")
    mark_warmup(axis)
    axis.set(title="a  Retrieved and accumulated state", xlabel="processed bases (millions)", ylabel="norm")
    axis.legend(frameon=False)
    style_axis(axis)

    axis = axes[0, 1]
    axis.plot(x, rolling(frame, "memory_update_norm"), color=colors["coral"], linewidth=1.8, label="memory update")
    surprise_axis = axis.twinx()
    surprise_axis.plot(x, rolling(frame, "surprise_norm"), color=colors["gold"], linewidth=1.6, label="surprise")
    mark_warmup(axis)
    axis.set(title="b  Write magnitude and surprise", xlabel="processed bases (millions)", ylabel="memory-update norm")
    surprise_axis.set_ylabel("surprise norm")
    lines = axis.lines[:1] + surprise_axis.lines
    axis.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8)
    style_axis(axis)
    surprise_axis.spines["top"].set_visible(False)

    axis = axes[1, 0]
    axis.plot(x, frame["alpha_mean"] * 1000, color=colors["coral"], linewidth=1.6, label="forget α ×1,000")
    axis.plot(x, frame["theta_mean"] * 1000, color=colors["blue"], linewidth=1.6, label="write θ ×1,000")
    retention_axis = axis.twinx()
    retention_axis.plot(x, frame["eta_mean"], color=colors["cyan"], linewidth=1.5, label="retention η")
    mark_warmup(axis)
    axis.set(title="c  Learned memory gates", xlabel="processed bases (millions)", ylabel="α and θ (×1,000)")
    retention_axis.set_ylabel("η")
    lines = axis.lines[:2] + retention_axis.lines
    axis.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8)
    style_axis(axis)
    retention_axis.spines["top"].set_visible(False)

    axis = axes[1, 1]
    early = frame.loc[frame["optimizer_step"] <= 160]
    axis.plot(early["optimizer_step"], early["memory_update_norm"], color=colors["coral"], linewidth=1.4, label="memory update")
    surprise_axis = axis.twinx()
    surprise_axis.plot(early["optimizer_step"], early["surprise_norm"], color=colors["gold"], linewidth=1.3, label="surprise")
    for step in (1, 33, 65, 97, 129):
        axis.axvline(step, color="#7B8794", linewidth=0.8, linestyle=":")
    axis.set(title="d  Early 96-segment rotation boundaries", xlabel="optimizer step", ylabel="memory-update norm")
    surprise_axis.set_ylabel("surprise norm")
    lines = axis.lines[:1] + surprise_axis.lines
    axis.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8)
    style_axis(axis)
    surprise_axis.spines["top"].set_visible(False)

    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def window_summary(frame: pd.DataFrame, start: int, end: int) -> dict[str, float | int]:
    selected = frame.loc[(frame["cumulative_bases"] > start) & (frame["cumulative_bases"] <= end)]
    keys = (
        "bits_per_base",
        "gradient_norm",
        "written_state_gradient_norm",
        "raw_memory_gradient_rms_max",
        "retrieval_norm",
        "memory_update_norm",
        "surprise_norm",
        "state_drift_norm",
        "past_momentary_cosine_mean",
        "alpha_mean",
        "eta_mean",
        "theta_mean",
        "token_accuracy",
        "top_2_accuracy",
        "bases_per_second",
    )
    return {"steps": int(len(selected)), **{key: float(selected[key].median()) for key in keys}}


def summarize(frame: pd.DataFrame, history: Path, live_status: Path) -> dict[str, object]:
    live = json.loads(live_status.read_text(encoding="utf-8"))
    run_state = str(live.get("state", "unknown"))
    bpb_500 = frame["bits_per_base"].rolling(500, min_periods=500).mean()
    best_index = int(bpb_500.idxmin())
    processed = int(frame["cumulative_bases"].iloc[-1])
    progress = float(live["progress_fraction"])
    # Drive may synchronize LIVE_STATUS.json and training_history.csv one step
    # apart. Derive the fixed panel target from internally consistent live-status
    # values, then use the newer history total for the remaining-time estimate.
    live_processed = int(live["processed_bases"])
    target = int(round(live_processed / progress)) if progress > 0 else 0
    recent_speed = float(frame["bases_per_second"].tail(500).mean())
    base_windows = {}
    for start in range(0, processed, 2_000_000):
        end = min(start + 2_000_000, processed)
        label = f"{start / 1e6:g}m_to_{end / 1e6:g}m"
        base_windows[label] = window_summary(frame, start, end)
    return {
        "format_version": 1,
        "snapshot_updated_at": live["updated_at"],
        "run_state": run_state,
        "history_sha256": sha256(history),
        "records": int(len(frame)),
        "optimizer_step": int(frame["optimizer_step"].iloc[-1]),
        "processed_bases": processed,
        "panel_target_bases_estimated_from_progress": target,
        "progress_fraction": progress,
        "recent_bases_per_second": recent_speed,
        "estimated_remaining_gpu_hours": (target - processed) / recent_speed / 3600,
        "all_numeric_values_finite": True,
        "interventions": {
            "memory_gradient_fraction_max": float(frame["memory_gradient_intervention_fraction"].max()),
            "legacy_surprise_fraction_max": float(frame["legacy_surprise_intervention_fraction"].max()),
            "memory_gradient_scale_min": float(frame["memory_gradient_scale_min"].min()),
        },
        "bpb": {
            "first_500_step_mean": float(frame["bits_per_base"].head(500).mean()),
            "latest_500_step_mean": float(frame["bits_per_base"].tail(500).mean()),
            "best_500_step_mean": float(bpb_500.iloc[best_index]),
            "best_500_step_ending_step": int(frame.loc[best_index, "optimizer_step"]),
            "best_500_step_ending_bases": int(frame.loc[best_index, "cumulative_bases"]),
        },
        "latest_500_step_means": {
            key: float(frame[key].tail(500).mean())
            for key in (
                "gradient_norm",
                "written_state_gradient_norm",
                "retrieval_norm",
                "memory_update_norm",
                "surprise_norm",
                "state_drift_norm",
                "raw_memory_gradient_rms_max",
                "past_momentary_cosine_mean",
                "token_accuracy",
                "top_2_accuracy",
                "alpha_mean",
                "eta_mean",
                "theta_mean",
            )
        },
        "base_windows": base_windows,
        "limits": [
            "BPB and accuracy are per-training-step telemetry, not held-out validation.",
            (
                "The run was complete when downloaded; downstream scientific gates are separate."
                if run_state == "completed"
                else "The run was active when downloaded; files are a time-stamped snapshot."
            ),
            "FAILED.txt must be interpreted against the ordered wrapper-step manifest, not in isolation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--live-status", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_history(args.history)
    plot_learning(frame, args.output_dir / "c19_learning_dynamics.png")
    plot_memory(frame, args.output_dir / "c19_memory_dynamics.png")
    summary = summarize(frame, args.history, args.live_status)
    (args.output_dir / "c19_snapshot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
