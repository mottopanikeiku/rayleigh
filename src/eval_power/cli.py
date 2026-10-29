"""Plan IID item counts using a fixed target difference and optional paired pilot."""

import argparse
import json
from pathlib import Path

import numpy as np

from .io import read_paired_csv
from .stats import (
    clustered_se,
    correlation,
    discordance_upper_bound,
    paired_variance,
    required_items,
    unpaired_variance,
)


def _plan(variance, args):
    """Items per model, or None: a zero plug-in variance is undefined, not zero items."""
    if variance == 0:
        return None
    return required_items(args.difference, variance, args.alpha, args.power)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--difference", type=float, required=True, help="fixed target accuracy gap, not a pilot gap"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--variance", type=float, help="positive variance of one item difference")
    source.add_argument("--pilot", help="shared-item CSV or binary NumPy .npy array, shape (n, 2)")
    parser.add_argument("--groups", help="aligned .npy group labels; only with --pilot")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    args = parser.parse_args(argv)
    if args.groups is not None and args.pilot is None:
        parser.error("--groups requires --pilot")

    result = {
        "difference": args.difference,
        "alpha": args.alpha,
        "target_power": args.power,
        "assumptions": [
            "Two-sided Gaussian approximation, not exact binary-test power.",
            "The target difference is fixed explicitly, not estimated from the pilot gap.",
            "The supplied or pilot-estimated variance transfers to the future evaluation.",
            "IID item sampling is assumed for item-count planning.",
            "Counts are items per model, not the total over both models.",
        ],
    }
    try:
        if args.pilot is None:
            result["variance"] = args.variance
            result["items_per_model"] = required_items(
                args.difference, args.variance, args.alpha, args.power
            )
        else:
            if Path(args.pilot).suffix.lower() == ".csv":
                _, pilot = read_paired_csv(args.pilot)
            else:
                pilot = np.load(args.pilot, allow_pickle=False)
            if not isinstance(pilot, np.ndarray) or pilot.ndim != 2 or pilot.shape[1] != 2:
                raise ValueError(
                    "pilot must be a shared-item CSV or binary .npy array, shape (n, 2)"
                )
            a, b = pilot[:, 0], pilot[:, 1]
            paired = paired_variance(a, b)
            unpaired = unpaired_variance(a, b)
            discordant = int(np.count_nonzero(a != b))
            bound = discordance_upper_bound(discordant, len(a))
            result.update(
                pilot_n=len(a),
                discordant_items=discordant,
                paired_variance=paired,
                discordance_upper_95=bound,
                unpaired_variance=unpaired,
                paired_correlation=correlation(a, b),
                paired_items_per_model=_plan(paired, args),
                bound_items_per_model=_plan(bound, args),
                unpaired_items_per_model=_plan(unpaired, args),
            )
            result["assumptions"].extend(
                [
                    "Paired planning uses same-item differences "
                    "and their sample variance (ddof=1); null if that variance is zero.",
                    "Bound planning replaces that variance with a one-sided 95% Clopper-Pearson "
                    "upper bound on P(A != B), which is at least Var(A-B). It ignores the pilot "
                    "gap and stays finite without discordant items; see docs/CALIBRATION.md.",
                    "Unpaired planning assumes independent model outcomes and binomial marginals.",
                ]
            )
            if args.groups is not None:
                groups = np.load(args.groups, allow_pickle=False)
                result["cluster_se"] = clustered_se(a, b, groups)
                result["group_count"] = len(np.unique(groups))
                result["assumptions"].append(
                    "Cluster SE is CR1 for the item-weighted gap and independent groups; "
                    "the reported IID item counts do not extrapolate cluster sampling. "
                    "More items within existing groups are not more independent groups."
                )
    except (OSError, ValueError, TypeError, EOFError, OverflowError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
