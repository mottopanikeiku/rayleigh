"""Regenerate the known-truth planning benchmark; CPU-only, no downloads."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import scipy

from eval_power.analysis import write_csv
from eval_power.calibration import (
    RULE_LABELS,
    RULES,
    SCENARIOS,
    confirmation_sensitivity,
    run_benchmark,
    scenario_label,
)

matplotlib.use("Agg")

SUMMARY = Path("results/prospective/summary.json")
PROTOCOL = Path("results/prospective/protocol.json")
COLORS = dict(zip(RULES, ["#a14c40", "#397698", "#478361", "#8a6d2f", "#555555"], strict=True))


def build(seed, outer, inner, trials, cap):
    """Simulation tables plus the measured sensitivity audit, with source hashes."""
    result = run_benchmark(seed, outer, inner, trials, cap)
    result["configuration"]["software"] = {"numpy": np.__version__, "scipy": scipy.__version__}
    sensitivity = confirmation_sensitivity(
        json.loads(SUMMARY.read_text()), json.loads(PROTOCOL.read_text())
    )
    sensitivity["sources"] = [
        {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (SUMMARY, PROTOCOL)
    ]
    return result, sensitivity


def make_figure(rows, destination):
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "rayleigh-known-truth"
    figure, axes = plt.subplots(
        1, len(SCENARIOS), figsize=(15, 4.8), sharey=True, layout="constrained"
    )
    for axis, (name, q, target) in zip(axes, SCENARIOS, strict=True):
        for rule in RULES:
            subset = [
                r
                for r in rows
                if r["scenario"] == name and r["rule"] == rule and r["delta"] == r["target"]
            ]
            x = [r["pilot_n"] for r in subset]
            y = np.array([r["nominal_workflow_rate"] for r in subset])
            low = np.array([r["nominal_workflow_mc_low"] for r in subset])
            high = np.array([r["nominal_workflow_mc_high"] for r in subset])
            axis.errorbar(
                x,
                y,
                yerr=[y - low, high - y],
                label=RULE_LABELS[rule],
                color=COLORS[rule],
                linestyle=":" if rule in ("known_variance", "exact_oracle") else "-",
                marker="o",
                capsize=3,
            )
        axis.axhline(0.8, linestyle="--", color="0.5", linewidth=1)
        axis.set_title(scenario_label(q, target), fontsize=10)
        axis.set_xscale("log", base=2)
        axis.set_xticks([32, 128, 512], ["32", "128", "512"])
        axis.set_xlabel("Pilot items")
        axis.set_ylim(0, 1.02)
    axes[0].set_ylabel("Rejection fraction at the meaningful effect")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=len(RULES), fontsize=9)
    figure.suptitle(
        "SIM: does an 80%-power plan deliver 80%? Exact McNemar test, nominal alpha=.05\n"
        "Unrun plans count as no detection; 95% Monte Carlo intervals cluster by pilot. "
        "Dotted rules know the true discordance q."
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination, metadata={"Date": None, "Title": "rayleigh: known-truth planning calibration"}
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--outer", type=int, default=1000)
    parser.add_argument("--inner", type=int, default=32)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--cap", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=Path("results/known_truth"))
    parser.add_argument("--figure", type=Path, default=Path("figures/known_truth.svg"))
    args = parser.parse_args()
    result, sensitivity = build(args.seed, args.outer, args.inner, args.trials, args.cap)
    args.output.mkdir(parents=True, exist_ok=True)
    for key in ("reference", "nested", "shared_models"):
        write_csv(args.output / f"{key}.csv", result[key])
    for key, document in (
        ("configuration", result["configuration"]),
        ("confirmation_sensitivity", sensitivity),
    ):
        (args.output / f"{key}.json").write_text(
            json.dumps(document, indent=2, allow_nan=False) + "\n"
        )
    make_figure(result["nested"], args.figure)
    holm = sensitivity["holm_detections"]
    print(
        f"SIM: {len(result['reference'])} exact-reference rows, "
        f"{len(result['nested'])} nested planning rows, "
        f"{len(result['shared_models'])} shared-model family rows; seed={args.seed}."
    )
    print(
        f"MEAS: {sensitivity['nominal_detections']}/{sensitivity['tested']} nominal; Holm "
        f"{holm['all_planned']} (all {sensitivity['planned']} planned), "
        f"{holm['executed_only']} (executed only), {holm['per_benchmark']} (per benchmark); "
        "median absolute realized gap="
        f"{100 * sensitivity['absolute_gap_quantiles']['median']:.1f}pp."
    )


if __name__ == "__main__":
    main()
