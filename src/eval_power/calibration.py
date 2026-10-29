"""Known-truth, IID binary-pair planning simulations (no model inference).

The meaningful effect is an input to the design, never inferred from a test set.
Outer replicates redraw pilots; inner replicates redraw independent heldout
families. Monte Carlo errors cluster at the outer replicate, not the inner test.
"""

import math

import numpy as np
from scipy.stats import beta, binom

from .analysis import wilson_interval
from .stats import _target_noncentrality, exact_mcnemar_power, holm_adjust, required_items

SCENARIOS = (
    ("rare_1pct", 0.01, 0.005),
    ("rare_5pct", 0.05, 0.02),
    ("close_2pp", 0.20, 0.02),
    ("moderate_10pp", 0.50, 0.10),
    ("small_n", 0.80, 0.60),
)
RULES = ("observed_gap", "fixed_effect", "variance_only_bound", "known_variance")
ALPHA = 0.05
POWER = 0.8
FAMILY = 3


def paired_probabilities(delta, discordance):
    """Cells [00, 01, 10, 11], with equal concordant probabilities."""
    if not np.isfinite(delta) or not abs(delta) <= discordance <= 1:
        raise ValueError("require finite abs(delta) <= discordance <= 1")
    return np.array(
        [
            (1 - discordance) / 2,
            (discordance - delta) / 2,
            (discordance + delta) / 2,
            (1 - discordance) / 2,
        ]
    )


def count_pvalues(wins, discordances):
    """Same exact conditional test as mcnemar_pvalue, in count space."""
    return np.minimum(1, 2 * binom.cdf(np.minimum(wins, discordances - wins), discordances, 0.5))


def family_adjust(pvalues):
    """Holm on the complete prespecified last-axis family, including unrun p=1."""
    order = np.argsort(pvalues, axis=-1, kind="stable")
    ranked = np.take_along_axis(pvalues, order, axis=-1)
    adjusted = np.minimum(1, np.maximum.accumulate(ranked * np.arange(FAMILY, 0, -1), axis=-1))
    return np.take_along_axis(adjusted, np.argsort(order, axis=-1), axis=-1)


def plan_counts(counts, target, discordance, rule):
    """Normal budgets: -1 denotes an undefined plan, never a zero-item study.

    fixed_effect uses unbiased pilot Var(A-B). variance_only_bound instead uses
    a one-sided 95% Clopper-Pearson upper bound on q=P(A!=B), hence on Var(A-B).
    It ignores the signed pilot gap, including in its variance estimate.
    known_variance uses true q-target**2 at the design alternative (an oracle).
    All rules target nominal, not familywise, 80% power.
    """
    counts = np.asarray(counts)
    n = counts.sum(axis=-1)
    if np.any(n < 2) or np.any(counts < 0):
        raise ValueError("pilots require at least two items and nonnegative counts")
    d = counts[..., 1] + counts[..., 2]
    gap = (counts[..., 2] - counts[..., 1]) / n
    variance = (d / n - gap**2) * n / (n - 1)
    effect = np.full(gap.shape, target)
    if rule == "observed_gap":
        effect = np.abs(gap)
    elif rule == "variance_only_bound":
        variance = np.ones(gap.shape)
        interior = d < n
        variance[interior] = beta.ppf(0.95, d[interior] + 1, n[interior] - d[interior])
    elif rule == "known_variance":
        variance = np.full(gap.shape, discordance - target**2)
    elif rule != "fixed_effect":
        raise ValueError(f"unknown planning rule: {rule}")
    valid = (variance > 0) & (effect > 0)
    result = np.full(gap.shape, -1, dtype=np.int64)
    factor = _target_noncentrality(ALPHA, POWER) ** 2
    result[valid] = np.maximum(1, np.ceil(factor * variance[valid] / effect[valid] ** 2)).astype(
        np.int64
    )
    return result


def clustered_rate(numerator, denominator=None):
    """Ratio of means and delta-method MC SE over independent outer replicates.

    Inputs have one value per outer replicate. This also handles the conditional
    executed-plan rate without pretending reused pilots are independent trials.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.ones_like(numerator) if denominator is None else np.asarray(denominator)
    if len(numerator) < 2:
        raise ValueError("at least two independent outer replicates required")
    scale = float(denominator.mean())
    if scale == 0:
        return {"rate": None, "mc_se": None, "mc_low": None, "mc_high": None}
    rate = float(numerator.mean() / scale)
    residual = numerator - rate * denominator
    se = float(residual.std(ddof=1) / (math.sqrt(len(residual)) * scale))
    return {
        "rate": rate,
        "mc_se": se,
        "mc_low": max(0, rate - 1.96 * se),
        "mc_high": min(1, rate + 1.96 * se),
    }


def nested_cell(delta, discordance, target, pilot_n, outer, inner, cap, seed):
    """Independent three-pair families, with new pilots in each outer replicate."""
    pilot_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    pilots = pilot_rng.multinomial(
        pilot_n, paired_probabilities(delta, discordance), size=(outer, FAMILY)
    )
    rows = []
    for index, rule in enumerate(RULES):
        plans = plan_counts(pilots, target, discordance, rule)
        executed = (plans > 0) & (plans <= cap)
        sizes = np.where(executed, plans, 0)
        rng = np.random.default_rng(np.random.SeedSequence([seed, index + 1]))
        m = rng.binomial(sizes[:, None, :], discordance, size=(outer, inner, FAMILY))
        wins = rng.binomial(m, (1 + delta / discordance) / 2)
        pvalues = count_pvalues(wins, m)
        nominal, holm = pvalues <= ALPHA, family_adjust(pvalues) <= ALPHA
        positive = plans[plans > 0]
        row = {
            "rule": rule,
            "outer_pilots": outer,
            "inner_families": inner,
            "family_size": FAMILY,
            "undefined_plans": int(np.sum(plans < 0)),
            "over_cap_plans": int(np.sum(plans > cap)),
            "executed_plans": int(executed.sum()),
            "total_plans": int(plans.size),
            "planned_n_p10": float(np.quantile(positive, 0.1)) if positive.size else None,
            "planned_n_median": float(np.median(positive)) if positive.size else None,
            "planned_n_p90": float(np.quantile(positive, 0.9)) if positive.size else None,
            "mean_executed_n": float(sizes.sum() / executed.sum()) if executed.any() else None,
        }
        for label, rejected in (("nominal", nominal), ("holm", holm)):
            rates = {
                "workflow": clustered_rate(rejected.mean(axis=(1, 2))),
                "conditional": clustered_rate(rejected.mean(axis=(1, 2)), executed.mean(axis=1)),
                "any": clustered_rate(rejected.any(axis=2).mean(axis=1)),
            }
            for scope, summary in rates.items():
                row.update({f"{label}_{scope}_{key}": value for key, value in summary.items()})
        rows.append(row)
    return rows


def reference_grid(seed, trials):
    """Exact summation versus independent fixed-N Monte Carlo, including nulls."""
    rows = []
    for scenario_index, (name, q, target) in enumerate(SCENARIOS):
        sizes = sorted({16, 32, 64, 128, 512, 2048, 8192, required_items(target, q - target**2)})
        for effect_index, multiplier in enumerate((0, 0.5, 1)):
            delta = target * multiplier
            for n in sizes:
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, 0, scenario_index, effect_index, n])
                )
                m = rng.binomial(n, q, trials)
                wins = rng.binomial(m, (1 + delta / q) / 2)
                successes = int(np.sum(count_pvalues(wins, m) <= ALPHA))
                low, high = wilson_interval(successes, trials)
                rows.append(
                    {
                        "scenario": name,
                        "delta": delta,
                        "discordance": q,
                        "n": n,
                        "exact_power": exact_mcnemar_power(delta, q, n),
                        "exact_bonferroni_power": exact_mcnemar_power(delta, q, n, ALPHA / FAMILY),
                        "trials": trials,
                        "detections": successes,
                        "mc_rate": successes / trials,
                        "mc_low": low,
                        "mc_high": high,
                    }
                )
    return rows


def shared_model_grid(seed, trials):
    """Three models share each item's latent difficulty; all three pairs tested.

    Half the items have correctness probabilities .2+offset, half .8+offset.
    Model draws are independent conditional on difficulty, not unconditionally.
    The mixed family has one true-null pair and two non-null pairs.
    """
    bits = ((np.arange(8)[:, None] >> np.arange(3)) & 1).astype(int)
    pairs = [(0, 1), (0, 2), (1, 2)]
    rows = []
    for design_index, (design, offset) in enumerate(
        (("global_null", np.zeros(3)), ("mixed", np.array([0.0, 0.0, 0.1])))
    ):
        probabilities = sum(
            np.prod(np.where(bits, base + offset, 1 - base - offset), axis=1) / 2
            for base in (0.2, 0.8)
        )
        nulls = np.array([offset[a] == offset[b] for a, b in pairs])
        for n in (16, 64, 256, 1024):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 2, design_index, n]))
            counts = rng.multinomial(n, probabilities, size=trials)
            pvalues = np.column_stack(
                [
                    count_pvalues(
                        counts @ ((bits[:, a] == 1) & (bits[:, b] == 0)),
                        counts @ (bits[:, a] != bits[:, b]),
                    )
                    for a, b in pairs
                ]
            )
            for label, values in (("nominal", pvalues), ("holm", family_adjust(pvalues))):
                # Independent families, not independent pairs: Wilson applies here.
                errors = int(np.sum((values[:, nulls] <= ALPHA).any(axis=1)))
                low, high = wilson_interval(errors, trials)
                rows.append(
                    {
                        "design": design,
                        "n": n,
                        "adjustment": label,
                        "trials": trials,
                        "false_positive_families": errors,
                        "fwer": errors / trials,
                        "mc_low": low,
                        "mc_high": high,
                    }
                )
    return rows


def run_benchmark(seed=20261029, outer=1000, inner=32, trials=20000, cap=20000):
    if min(outer, inner, trials, cap) < 2 or seed < 0:
        raise ValueError("replicate counts and cap must be >=2, seed must be nonnegative")
    nested = []
    for scenario_index, (name, q, target) in enumerate(SCENARIOS):
        for effect_index, multiplier in enumerate((0, 0.5, 1)):
            for pilot_n in (32, 128, 512):
                cell_seed = int(
                    np.random.SeedSequence(
                        [seed, 1, scenario_index, effect_index, pilot_n]
                    ).generate_state(1)[0]
                )
                identity = {
                    "scenario": name,
                    "delta": target * multiplier,
                    "discordance": q,
                    "target": target,
                    "pilot_n": pilot_n,
                    "seed": cell_seed,
                }
                nested.extend(
                    {**identity, **row}
                    for row in nested_cell(
                        target * multiplier, q, target, pilot_n, outer, inner, cap, cell_seed
                    )
                )
    return {
        "configuration": {
            "evidence": "SIM: known-truth simulation, not model evaluations",
            "seed": seed,
            "rng": "NumPy PCG64 via SeedSequence",
            "outer": outer,
            "inner": inner,
            "reference_trials": trials,
            "cap": cap,
            "alpha": ALPHA,
            "target_power": POWER,
            "family_size": FAMILY,
            "pilot_sizes": [32, 128, 512],
            "true_effect_multipliers": [0, 0.5, 1],
            "scenarios": [
                {"name": name, "discordance": q, "meaningful_effect": target}
                for name, q, target in SCENARIOS
            ],
            "seed_recipe": "reference [seed,0,scenario,effect,n]; nested cell "
            "SeedSequence([seed,1,scenario,effect,pilot_n]).generate_state(1)[0], "
            "then [cell,0] pilots / [cell,rule_index+1] heldout; "
            "shared models [seed,2,design,n]",
        },
        "reference": reference_grid(seed, trials),
        "nested": nested,
        "shared_models": shared_model_grid(seed, trials),
    }


def confirmation_sensitivity(summary):
    """Descriptive multiplicity/effect-size audit of committed repeated-decoding data."""
    confirmations = [
        row["confirmation"] for row in summary["pairs"] if row["confirmation"] is not None
    ]
    pvalues = [row["pvalue"] for row in confirmations]
    gaps = np.abs([row["mean_difference"] for row in confirmations])
    return {
        "evidence": "MEAS: committed fresh-item confirmations, not simulation",
        "tested": len(pvalues),
        "nominal_detections": int(np.sum(np.array(pvalues) < ALPHA)),
        "holm_detections": int(np.sum(holm_adjust(pvalues) < ALPHA)),
        "family": "All executed primary confirmations across both benchmarks; unrun plans "
        "are not part of this descriptive 27-test sensitivity family.",
        "absolute_gap_quantiles": dict(
            zip(
                ("min", "q25", "median", "q75", "max"),
                np.quantile(gaps, [0, 0.25, 0.5, 0.75, 1]),
                strict=True,
            )
        ),
        "gaps_at_most_1pp": int(np.sum(gaps <= 0.01)),
        "n16_comparisons": sum(row["n_items"] == 16 for row in confirmations),
        "holm_adjusted_pvalues": holm_adjust(pvalues).tolist(),
    }
