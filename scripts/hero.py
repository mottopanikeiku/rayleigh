"""Draw the README hero, light and dark, from committed result files only."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from eval_power.calibration import POWER, RULE_LABELS, RULES, SCENARIOS

NESTED = Path("results/known_truth/nested.csv")
SENSITIVITY = Path("results/known_truth/confirmation_sensitivity.json")
PILOT_N = 128
THEMES = {
    "light": {
        "background": "#f6f4ef",
        "edge": "#dcd8ce",
        "text": "#1c1b18",
        "muted": "#69655d",
        "cells": ("#d5e6d8", "#f1e0b8", "#efcdc5"),
    },
    "dark": {
        "background": "#161b22",
        "edge": "#30363d",
        "text": "#e6edf3",
        "muted": "#9aa4ae",
        "cells": ("#1e3d2a", "#4a3b16", "#4f231d"),
    },
}


def hero_cells():
    """Nominal workflow rejection and its MC upper limit at the meaningful effect."""
    with NESTED.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["scenario"], row["rule"]): (
            float(row["nominal_workflow_rate"]),
            float(row["nominal_workflow_mc_high"]),
        )
        for row in rows
        if int(row["pilot_n"]) == PILOT_N and row["delta"] == row["target"]
    }


def make_hero(theme, destination):
    colors = THEMES[theme]
    cells = hero_cells()
    sensitivity = json.loads(SENSITIVITY.read_text())
    holm = sensitivity["holm_detections"]
    median_gap = sensitivity["absolute_gap_quantiles"]["median"]
    plt.rcParams.update({"svg.hashsalt": f"rayleigh-hero-{theme}", "font.family": "DejaVu Sans"})
    figure = plt.figure(figsize=(8, 4.1), facecolor=colors["background"])
    figure.patches.append(
        FancyBboxPatch(
            (0.004, 0.006),
            0.992,
            0.988,
            boxstyle="round,pad=0,rounding_size=0.02",
            transform=figure.transFigure,
            facecolor=colors["background"],
            edgecolor=colors["edge"],
            linewidth=1,
            zorder=-1,
        )
    )
    lines = (
        (0.9, 16, "text", "Which 80%-power plan delivers 80%?"),
        (
            0.835,
            8.5,
            "muted",
            f"SIM, known truth: {PILOT_N}-item pilot, exact McNemar test at the meaningful "
            "effect, unrun plans count as misses",
        ),
        (
            0.115,
            8,
            "muted",
            "Green: the 95% Monte Carlo interval reaches 80%. Amber: within 10 points. "
            "Red: further short.",
        ),
        (
            0.045,
            8.5,
            "text",
            f"MEAS, six open models: {sensitivity['nominal_detections']}/{sensitivity['tested']}"
            f" fresh comparisons detected, {holm['all_planned']} after Holm over all "
            f"{sensitivity['planned']} planned; median realized gap {100 * median_gap:.1f} points.",
        ),
    )
    for y, size, color, line in lines:
        figure.text(
            0.035,
            y,
            line,
            fontsize=size,
            fontweight="bold" if size > 10 else "normal",
            color=colors[color],
            transform=figure.transFigure,
        )
    axis = figure.add_axes((0.36, 0.19, 0.615, 0.54))
    axis.set_xlim(0, len(SCENARIOS))
    axis.set_ylim(len(RULES), 0)
    axis.axis("off")
    for column, (name, q, target) in enumerate(SCENARIOS):
        axis.text(
            column + 0.5,
            -0.12,
            f"q={100 * q:g}%\n{100 * target:g}-pt effect",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=colors["muted"],
        )
        for row, rule in enumerate(RULES):
            rate, high = cells[name, rule]
            level = 0 if high >= POWER else 1 if rate >= POWER - 0.1 else 2
            axis.add_patch(
                Rectangle(
                    (column + 0.04, row + 0.06),
                    0.92,
                    0.88,
                    facecolor=colors["cells"][level],
                    edgecolor="none",
                )
            )
            axis.text(
                column + 0.5,
                row + 0.5,
                f"{100 * rate:.0f}%",
                ha="center",
                va="center",
                fontsize=11,
                color=colors["text"],
            )
    for row, rule in enumerate(RULES):
        axis.text(
            -0.08,
            row + 0.5,
            RULE_LABELS[rule],
            ha="right",
            va="center",
            fontsize=9.5,
            color=colors["text"],
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        metadata={"Date": None, "Title": "rayleigh: which 80%-power plan delivers 80%?"},
        facecolor=colors["background"],
    )
    plt.close(figure)


def main():
    for theme in THEMES:
        make_hero(theme, Path("docs") / f"hero-{theme}.svg")
    print("Wrote docs/hero-light.svg and docs/hero-dark.svg.")


if __name__ == "__main__":
    main()
