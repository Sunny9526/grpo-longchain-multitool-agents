"""Render README ablation figures from the documented result summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = REPO_ROOT / "docs/experiments/ablation/results_summary.csv"
DEFAULT_OUTPUT = REPO_ROOT / "docs/assets/figures"

METHOD_ORDER = [
    "Vanilla",
    "Turn-Discount",
    "PRM-Lite",
    "LATA",
    "PRM-Lite + LATA",
]
COLORS = {
    "Vanilla": "#7A7A7A",
    "Turn-Discount": "#E69F00",
    "PRM-Lite": "#CC79A7",
    "LATA": "#56B4E9",
    "PRM-Lite + LATA": "#009E73",
}
LABELS = {
    "Vanilla": "Vanilla",
    "Turn-Discount": "Turn-Discount",
    "PRM-Lite": "PRM-Lite",
    "LATA": "LATA",
    "PRM-Lite + LATA": "Joint",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str | float | int]]:
    numeric_float = {
        "overall_pass1",
        "generalization_pass1",
        "covered_pass1",
        "uncovered_pass1",
        "unseen_pass1",
        "error_rate",
    }
    rows: list[dict[str, str | float | int]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, str | float | int] = dict(raw)
            row["step"] = int(raw["step"])
            row["per_turn_p50_tokens"] = int(raw["per_turn_p50_tokens"])
            for key in numeric_float:
                row[key] = float(raw[key])
            rows.append(row)
    return rows


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
        }
    )


def selected_rows(rows: list[dict[str, str | float | int]]) -> list[dict[str, str | float | int]]:
    chosen: dict[str, dict[str, str | float | int]] = {}
    for row in rows:
        if row["selection"] in {"selected", "ablation_baseline"}:
            chosen[str(row["experiment"])] = row
    missing = set(METHOD_ORDER) - set(chosen)
    if missing:
        raise ValueError(f"Missing selected comparison rows: {sorted(missing)}")
    return [chosen[method] for method in METHOD_ORDER]


def plot_comparison(rows: list[dict[str, str | float | int]], output_dir: Path) -> None:
    chosen = selected_rows(rows)
    panels = [
        ("overall_pass1", "Overall pass$^1$", "Score", (0.0, 0.28), "{:.3f}"),
        ("generalization_pass1", "Generalization pass$^1$", "Score", (0.0, 0.13), "{:.3f}"),
        ("error_rate", "Tool error rate", "Rate (lower is better)", (0.0, 0.40), "{:.3f}"),
        ("per_turn_p50_tokens", "Per-turn reasoning depth", "Tokens (p50)", (0, 350), "{:.0f}"),
    ]
    x = range(len(chosen))
    colors = [COLORS[str(row["experiment"])] for row in chosen]

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.5))
    for panel_label, (axis, (key, title, ylabel, ylim, value_format)) in zip(
        ("a", "b", "c", "d"), zip(axes.flat, panels)
    ):
        values = [float(row[key]) for row in chosen]
        bars = axis.bar(x, values, color=colors, width=0.68)
        axis.set_title(title, pad=8)
        axis.set_ylabel(ylabel)
        axis.set_ylim(*ylim)
        axis.set_xticks(list(x), [LABELS[str(row["experiment"])] for row in chosen], rotation=18, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.text(-0.12, 1.06, f"({panel_label})", transform=axis.transAxes, weight="bold")
        padding = (ylim[1] - ylim[0]) * 0.025
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + padding,
                value_format.format(value),
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    fig.suptitle("Ablation results at selected checkpoints", fontsize=12, weight="semibold", y=1.01)
    fig.text(
        0.5,
        -0.02,
        "Source: documented independent-evaluation summary (N=4 per task, max_tokens=4096).",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "ablation_comparison")


def plot_progression(rows: list[dict[str, str | float | int]], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 4.7))
    for method in METHOD_ORDER:
        method_rows = sorted(
            [row for row in rows if row["experiment"] == method],
            key=lambda row: int(row["step"]),
        )
        steps = [int(row["step"]) for row in method_rows]
        values = [float(row["overall_pass1"]) for row in method_rows]
        axis.plot(
            steps,
            values,
            color=COLORS[method],
            linewidth=2.0 if method == "PRM-Lite + LATA" else 1.5,
            marker="o",
            markersize=5,
            label=LABELS[method],
        )
        for row in method_rows:
            if row["status"] == "reported_estimate":
                axis.scatter(
                    int(row["step"]),
                    float(row["overall_pass1"]),
                    s=55,
                    facecolors="white",
                    edgecolors=COLORS[method],
                    linewidths=1.5,
                    zorder=4,
                )

    axis.set_title("Independent evaluation across training steps", pad=10)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Overall pass$^1$")
    axis.set_xlim(40, 310)
    axis.set_ylim(0.07, 0.26)
    axis.xaxis.set_major_locator(MultipleLocator(50))
    axis.yaxis.set_major_locator(MultipleLocator(0.05))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(ncol=3, frameon=False, loc="lower right")
    fig.text(
        0.5,
        0.01,
        "Open marker: value reported in the diagnosis table but marked estimated by the legacy plotting script.",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, output_dir, "ablation_progression")


def main() -> None:
    args = parse_args()
    configure_style()
    rows = load_rows(args.data)
    plot_comparison(rows, args.output_dir)
    plot_progression(rows, args.output_dir)
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
