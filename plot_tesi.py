#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

TITLE_FONT_SIZE = 22
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 17
TICK_STEP_METERS = 5


def load_sequence(data_dir: Path, seq_id: int):
    seq_dir = data_dir / f"seq{seq_id}"
    gt_path = seq_dir / "gt.csv"
    mic_path = seq_dir / "microphones.csv"
    speech_path = seq_dir / "speech_trajectory.csv"
    params_path = seq_dir / "params.json"

    if not gt_path.exists():
        raise FileNotFoundError(f"Missing file: {gt_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"Missing file: {params_path}")

    df = pd.read_csv(gt_path)
    required_columns = ["sx", "sy", "is_active"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"{gt_path} must contain columns {required_columns}")

    df_mics = pd.read_csv(mic_path) if mic_path.exists() else None
    df_speech = pd.read_csv(speech_path) if speech_path.exists() else None

    with params_path.open("r", encoding="utf-8") as f:
        params = json.load(f)

    scenario = params.get("scenario", f"seq{seq_id}")
    return df, df_mics, df_speech, scenario


def plot_single_axis(ax, df, df_mics, df_speech, title, xlim=None, ylim=None):
    ax.plot(df["sx"], df["sy"], color="gray", linestyle="-", linewidth=1, alpha=0.3)

    for i in range(len(df) - 1):
        x_vals = [df["sx"].iloc[i], df["sx"].iloc[i + 1]]
        y_vals = [df["sy"].iloc[i], df["sy"].iloc[i + 1]]

        if df["is_active"].iloc[i] == 1:
            ax.plot(x_vals, y_vals, color="red", linestyle="-", linewidth=2)
        else:
            ax.plot(x_vals, y_vals, color="gray", linestyle="--", linewidth=2)

    ax.scatter(df["sx"].iloc[0], df["sy"].iloc[0], color="green", s=100, marker="s", zorder=5)
    ax.scatter(df["sx"].iloc[-1], df["sy"].iloc[-1], color="purple", s=100, marker="X", zorder=5)

    if df_speech is not None and all(col in df_speech.columns for col in ["px", "py"]):
        ax.plot(
            df_speech["px"],
            df_speech["py"],
            color="orange",
            linestyle="-",
            linewidth=1.2,
            alpha=0.6,
        )
        ax.scatter(df_speech["px"].iloc[0], df_speech["py"].iloc[0], color="orange", s=60, marker="s", zorder=5)
        ax.scatter(df_speech["px"].iloc[-1], df_speech["py"].iloc[-1], color="orange", s=60, marker="X", zorder=5)

    if df_mics is not None and all(col in df_mics.columns for col in ["mx", "my"]):
        ax.scatter(df_mics["mx"], df_mics["my"], color="blue", s=80, marker="o", zorder=6)
    else:
        ax.scatter(0, 0, color="blue", s=150, marker="o", zorder=6)

    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.xaxis.set_major_locator(MultipleLocator(TICK_STEP_METERS))
    ax.yaxis.set_major_locator(MultipleLocator(TICK_STEP_METERS))
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.axis("equal")


def main():
    parser = argparse.ArgumentParser(
        description="Plot selected trajectories in separate subplots using the same style as plot_trajectory.py."
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Path to the data directory containing seq*/ folders (default: data)",
    )
    parser.add_argument(
        "--output",
        default="tesi_trajectories.png",
        help="Output image path (default: tesi_trajectories.png)",
    )
    parser.add_argument(
        "--seq_ids",
        nargs="+",
        type=int,
        default=[2, 1, 5, 3, 8],
        help="Sequence IDs to plot (default: 2 1 5 3 8)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    n = len(args.seq_ids)

    if n == 5:
        # Dedicated 2-2-1 layout: last subplot spans both columns.
        fig = plt.figure(figsize=(14, 15))
        gs = fig.add_gridspec(3, 2)
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, :]),
        ]
    else:
        cols = 2
        rows = (n + cols - 1) // cols
        fig, axes_grid = plt.subplots(rows, cols, figsize=(14, 5.5 * rows), squeeze=False)
        axes = list(axes_grid.ravel())

    sequences = []
    for seq_id in args.seq_ids:
        df, df_mics, df_speech, scenario = load_sequence(data_dir, seq_id)
        sequences.append((seq_id, df, df_mics, df_speech, scenario))

    all_x_min = min(df["sx"].min() for _, df, _, _, _ in sequences)
    all_x_max = max(df["sx"].max() for _, df, _, _, _ in sequences)
    all_y_min = min(df["sy"].min() for _, df, _, _, _ in sequences)
    all_y_max = max(df["sy"].max() for _, df, _, _, _ in sequences)

    xlim = (
        math.floor(all_x_min / TICK_STEP_METERS) * TICK_STEP_METERS,
        math.ceil(all_x_max / TICK_STEP_METERS) * TICK_STEP_METERS,
    )
    ylim = (
        math.floor(all_y_min / TICK_STEP_METERS) * TICK_STEP_METERS,
        math.ceil(all_y_max / TICK_STEP_METERS) * TICK_STEP_METERS,
    )

    for idx, (_, df, df_mics, df_speech, scenario) in enumerate(sequences):
        title = scenario.replace("_", " ")
        plot_single_axis(axes[idx], df, df_mics, df_speech, title, xlim=xlim, ylim=ylim)

    legend_handles = [
        plt.Line2D([0], [0], color="red", linestyle="-", linewidth=2, label="Siren ON"),
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label="Siren OFF"),
        plt.Line2D([0], [0], color="green", marker="s", linestyle="None", markersize=10, label="Start"),
        plt.Line2D([0], [0], color="purple", marker="X", linestyle="None", markersize=10, label="End"),
        plt.Line2D([0], [0], color="orange", linestyle="-", linewidth=1.2, label="Speech walker"),
        plt.Line2D([0], [0], color="blue", marker="o", linestyle="None", markersize=8, label="Microphones"),
    ]

    if len(axes) > n:
        for idx in range(n, len(axes)):
            axes[idx].axis("off")

    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        ncol=6,
        fontsize=LEGEND_FONT_SIZE,
        markerscale=1.3,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
