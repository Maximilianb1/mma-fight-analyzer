"""Generate the report's evidence-backed figures from the checked-in result tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


REPORT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPORT_DIR / "data"
FIGURE_DIR = REPORT_DIR / "figures"

NAVY = "#17324D"
BLUE = "#3D6D9A"
TEAL = "#2A9D8F"
ORANGE = "#E88C3A"
LIGHT_BLUE = "#A9C5DA"
GRAY = "#7A8793"


def setup_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.titleweight": "bold",
            "figure.dpi": 180,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def save(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / f"{name}.pdf",
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    png_path = FIGURE_DIR / f"{name}.png"
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    # Matplotlib writes RGBA PNGs even with an opaque background.  Converting
    # to RGB avoids black transparency regions in some PDF renderers.
    with Image.open(png_path) as image:
        image.convert("RGB").save(png_path)
    plt.close(fig)


def dataset_distribution():
    data = pd.read_csv(DATA_DIR / "dataset_counts.csv")
    fig, axes = plt.subplots(
        1, 3, figsize=(7.15, 2.25), gridspec_kw={"width_ratios": [1.55, 1.1, 0.8]}
    )
    palettes = {
        "Phase": [NAVY, BLUE, TEAL, ORANGE, LIGHT_BLUE],
        "Pressure": [TEAL, ORANGE, BLUE],
        "Segment": [NAVY, GRAY],
    }
    for axis, group in zip(axes, ["Phase", "Pressure", "Segment"]):
        subset = data[data.group == group]
        bars = axis.barh(
            subset.label,
            subset["count"],
            color=palettes[group][: len(subset)],
            edgecolor="white",
        )
        axis.invert_yaxis()
        axis.set_title(group)
        axis.set_xlabel("Clips")
        axis.grid(axis="x", alpha=0.25)
        axis.grid(axis="y", visible=False)
        axis.bar_label(bars, padding=2, fontsize=7.5)
        axis.set_xlim(0, subset["count"].max() * 1.18)
        sns.despine(ax=axis, left=True, bottom=True)
    fig.suptitle("Dataset composition (1,315 clips from 11 fights)", y=1.03, weight="bold")
    fig.tight_layout(w_pad=1.1)
    save(fig, "dataset_distribution")


def model_comparison():
    data = pd.read_csv(DATA_DIR / "experiment_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8), sharex=True)
    configurations = [
        ("phase_macro_f1", "Phase macro-F1", "R(2+1)D multi-task"),
        ("pressure_macro_f1", "Pressure macro-F1", "R(2+1)D pressure-only"),
    ]
    for axis, (column, title, selected) in zip(axes, configurations):
        subset = data.dropna(subset=[column]).sort_values(column)
        colors = [ORANGE if name == selected else BLUE for name in subset.model]
        bars = axis.barh(subset.model, subset[column], color=colors)
        axis.set_title(title)
        axis.set_xlim(0.35, 0.69)
        axis.set_xlabel("Macro-F1")
        axis.grid(axis="x", alpha=0.25)
        axis.grid(axis="y", visible=False)
        axis.bar_label(bars, labels=[f"{v:.3f}" for v in subset[column]], padding=2, fontsize=7)
        sns.despine(ax=axis, left=True, bottom=True)
    fig.suptitle("Final five-fold development comparison", y=1.02, weight="bold")
    fig.tight_layout(w_pad=1.0)
    save(fig, "model_comparison")


def confusion_axis(axis, csv_name, title):
    frame = pd.read_csv(DATA_DIR / csv_name).set_index("true_label")
    matrix = frame.to_numpy()
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    sns.heatmap(
        normalized,
        annot=False,
        cmap=sns.light_palette(BLUE, as_cmap=True),
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=0.6,
        linecolor="white",
        xticklabels=frame.columns,
        yticklabels=frame.index,
        square=True,
        ax=axis,
    )
    # Add the integer counts explicitly.  This also gives deterministic
    # high-contrast text when the normalized cell color is dark.
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if normalized[row, column] >= 0.55 else "#222222"
            axis.text(
                column + 0.5,
                row + 0.5,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.tick_params(axis="x", rotation=35, labelsize=7)
    axis.tick_params(axis="y", rotation=0, labelsize=7)


def holdout_confusions():
    fig, axes = plt.subplots(
        1, 2, figsize=(7.15, 2.95), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    confusion_axis(
        axes[0], "holdout_phase_confusion.csv", "Phase: 72.4% accuracy, 0.495 macro-F1"
    )
    confusion_axis(
        axes[1],
        "holdout_pressure_confusion.csv",
        "Pressure: 35.3% accuracy, 0.327 macro-F1",
    )
    fig.suptitle("Untouched-fight confusion matrices", y=1.02, weight="bold")
    fig.tight_layout(w_pad=1.2)
    save(fig, "holdout_confusions")


def main():
    setup_style()
    dataset_distribution()
    model_comparison()
    holdout_confusions()
    print("Figures written to report/figures")


if __name__ == "__main__":
    main()
