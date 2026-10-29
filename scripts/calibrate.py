"""Regenerate the known-truth planning benchmark; CPU-only, no downloads."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import scipy

from eval_power.analysis import write_csv
from eval_power.calibration import RULES, SCENARIOS, confirmation_sensitivity, run_benchmark

matplotlib.use("Agg")


def make_figure(rows, destination):
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "rayleigh-known-truth"
    colors = ["#a14c40", "#397698", "#478361", "#666666"]
    figure, axes = plt.subplots(
        1, len(SCENARIOS), figsize=(15, 4.5), sharey=True, layout="constrained"
    )
    for axis, (name, _, _) in zip(axes, SCENARIOS, strict=True):
        for rule, color in zip(RULES, colors, strict=True):
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
                label=rule.replace("_", " "),
                color=color,
                marker="o",
                capsize=3,
            )
        axis.axhline(0.8, linestyle="--", color="0.5", linewidth=1)
        axis.set_title(name.replace("_", " "))
        axis.set_xscale("log", base=2)
        axis.set_xticks([32, 128, 512], ["32", "128", "512"])
        axis.set_xlabel("Pilot items")
        axis.set_ylim(0, 1.02)
    axes[0].set_ylabel("Nominal workflow rejection fraction")
    axes[-1].legend(fontsize=8, loc="lower right")
    figure.suptitle(
        "SIM: known-truth planning at the meaningful effect\n"
        "Unrun plans count as no detection; 95% Monte Carlo intervals cluster by pilot"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, metadata={"Date": None})
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
    result = run_benchmark(args.seed, args.outer, args.inner, args.trials, args.cap)
    source = Path("results/prospective/summary.json")
    sensitivity = confirmation_sensitivity(json.loads(source.read_text()))
    sensitivity["source"] = {
        "file": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    result["configuration"]["software"] = {"numpy": np.__version__, "scipy": scipy.__version__}
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
    print(
        f"SIM: {len(result['reference'])} exact-reference rows, "
        f"{len(result['nested'])} nested planning rows, "
        f"{len(result['shared_models'])} shared-model family rows; seed={args.seed}."
    )
    print(
        f"MEAS: {sensitivity['nominal_detections']}/{sensitivity['tested']} nominal, "
        f"{sensitivity['holm_detections']}/{sensitivity['tested']} Holm; "
        "median absolute realized gap="
        f"{100 * sensitivity['absolute_gap_quantiles']['median']:.1f}pp."
    )


if __name__ == "__main__":
    main()
