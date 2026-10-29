"""Known-truth, IID binary-pair planning simulations (no model inference).

The meaningful effect is an input to the design, never inferred from a test set.
Outer replicates redraw pilots; inner replicates redraw independent heldout
families. Monte Carlo errors cluster at the outer replicate, not the inner test.
"""

import math
from functools import cache

import numpy as np
from scipy.stats import binom, nct, norm, t

from .analysis import wilson_interval
from .stats import (
    _mcnemar_tails,
    _normal_power,
    _target_noncentrality,
    discordance_upper_bound,
    exact_mcnemar_power,
    holm_adjust,
    required_items,
)

SCENARIOS = (
    ("rare_1pct", 0.01, 0.005),
    ("rare_5pct", 0.05, 0.02),
    ("close_2pp", 0.20, 0.02),
    ("moderate_10pp", 0.50, 0.10),
    ("small_n", 0.80, 0.60),
)
RULES = ("observed_gap", "fixed_effect", "discordance_bound", "known_variance", "exact_oracle")
RULE_LABELS = {
    "observed_gap": "Observed pilot gap",
    "fixed_effect": "Fixed effect, pilot variance",
    "discordance_bound": "Fixed effect, discordance bound",
    "known_variance": "Fixed effect, true variance (Gaussian)",
    "exact_oracle": "Exact oracle (true q, exact test)",
}
ALPHA = 0.05
POWER = 0.8
FAMILY = 3


def scenario_label(discordance, target):
    """Display name for a scenario: discordance rate and meaningful effect."""
    return f"q={100 * discordance:g}%, {100 * target:g}-point effect"


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


@cache
def exact_oracle_items(target, discordance):
    """Smallest N whose exact two-sided McNemar power at the true DGP reaches POWER.

    Exact power is sawtoothed in N, so a slightly larger N can fall back below
    the target. The linear search stops at four times the Gaussian budget at the
    null variance q, plus 64 items; no grid scenario comes close to that limit.
    """
    if not 0 < abs(target) <= discordance <= 1:
        raise ValueError("require 0 < abs(target) <= discordance <= 1")
    limit = 4 * required_items(target, discordance, ALPHA, POWER) + 64
    m = np.arange(limit + 1)
    low, high = _mcnemar_tails(m, (1 + target / discordance) / 2, ALPHA)
    conditional = low + high
    for n in range(1, limit + 1):
        if binom.pmf(m[: n + 1], n, discordance) @ conditional[: n + 1] >= POWER:
            return n
    raise ValueError("exact oracle budget exceeds the search limit")


def exact_wrong_sign(delta, discordance, n):
    """Exact probability that the two-sided test rejects in the wrong direction."""
    if delta == 0:
        return None
    m = np.arange(n + 1)
    low, high = _mcnemar_tails(m, (1 + delta / discordance) / 2, ALPHA)
    return float(binom.pmf(m, n, discordance) @ (low if delta > 0 else high))


def known_truth_budgets(target, discordance):
    """Three known-q budgets and their exact power: where Gaussian planning loses power.

    The Wald budget uses Var(A-B)=q-target**2 for both normal quantiles. The
    McNemar statistic's null variance is q, so the null-variance budget uses q
    for the alpha quantile and q-target**2 for the power quantile. The exact
    oracle searches exact power directly.
    """
    alternative = discordance - target**2
    null_variance = math.ceil(
        (
            (
                norm.ppf(1 - ALPHA / 2) * math.sqrt(discordance)
                + norm.ppf(POWER) * math.sqrt(alternative)
            )
            / target
        )
        ** 2
    )
    budgets = {
        "gaussian_known_variance": required_items(target, alternative, ALPHA, POWER),
        "null_variance_gaussian": null_variance,
        "exact_oracle": exact_oracle_items(target, discordance),
    }
    result = {}
    for name, n in budgets.items():
        result[f"{name}_n"] = n
        result[f"{name}_exact_power"] = exact_mcnemar_power(target, discordance, n)
    return result


def plan_counts(counts, target, discordance, rule):
    """Per-pilot budgets: -1 denotes an undefined plan, never a zero-item study.

    fixed_effect uses unbiased pilot Var(A-B) in the Gaussian formula.
    discordance_bound instead uses a one-sided 95% Clopper-Pearson upper bound
    on q=P(A!=B), hence on Var(A-B); it ignores the signed pilot gap.
    known_variance uses true q-target**2 at the design alternative in the same
    Gaussian formula (a Wald-test oracle). exact_oracle is the smallest N with
    exact McNemar power >= POWER at the true q and the meaningful effect.
    All rules target nominal, not familywise, 80% power.
    """
    counts = np.asarray(counts)
    n = counts.sum(axis=-1)
    if np.any(n < 2) or np.any(counts < 0):
        raise ValueError("pilots require at least two items and nonnegative counts")
    d = counts[..., 1] + counts[..., 2]
    gap = (counts[..., 2] - counts[..., 1]) / n
    if rule == "exact_oracle":
        return np.full(gap.shape, exact_oracle_items(target, discordance), dtype=np.int64)
    variance = (d / n - gap**2) * n / (n - 1)
    effect = np.full(gap.shape, target)
    if rule == "observed_gap":
        effect = np.abs(gap)
    elif rule == "discordance_bound":
        variance = discordance_upper_bound(d, n)
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
        # A two-sided rejection that points away from the true sign is a type S error.
        wrong = None
        if delta != 0:
            wrong = clustered_rate(
                (nominal & (np.sign(2 * wins - m) == -np.sign(delta))).mean(axis=(1, 2))
            )
        row["nominal_wrong_sign_rate"] = None if wrong is None else wrong["rate"]
        row["nominal_wrong_sign_mc_se"] = None if wrong is None else wrong["mc_se"]
        rows.append(row)
    return rows


def reference_grid(seed, trials):
    """Exact summation versus independent fixed-N Monte Carlo, including nulls."""
    rows = []
    for scenario_index, (name, q, target) in enumerate(SCENARIOS):
        sizes = sorted(
            {
                16,
                32,
                64,
                128,
                512,
                2048,
                8192,
                required_items(target, q - target**2),
                exact_oracle_items(target, q),
            }
        )
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
                        "exact_wrong_sign": exact_wrong_sign(delta, q, n),
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
            "rules": list(RULES),
            "scenarios": [
                {
                    "name": name,
                    "discordance": q,
                    "meaningful_effect": target,
                    **known_truth_budgets(target, q),
                }
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


def _holm_by_family(pvalues, families):
    """Holm-adjusted p-values with one complete family per distinct label."""
    adjusted = np.empty_like(pvalues)
    for family in np.unique(families):
        adjusted[families == family] = holm_adjust(pvalues[families == family])
    return adjusted


def confirmation_sensitivity(summary, protocol):
    """Multiplicity, effect-size and t-reference audit of committed repeated-decoding data.

    Detection is the experiment's prespecified rule: a two-sided paired t-test on
    item means with p < alpha and no multiplicity correction. The Holm families
    are post hoc sensitivity definitions; unrun (over-limit) plans enter a planned
    family with p=1. Plug-in power uses each pilot's own gap and planning
    variance at the planned item count: it restates the plan's assumptions under
    the t reference distribution, not the comparison's true power.
    """
    alpha = protocol["alpha"]
    floor = protocol["minimum_confirm_items"]
    pairs = summary["pairs"]
    ran = np.array([row["confirmation"] is not None for row in pairs])
    pvalues = np.array(
        [row["confirmation"]["pvalue"] if row["confirmation"] else 1.0 for row in pairs]
    )
    benchmarks = np.array([row["benchmark"] for row in pairs])
    adjusted = {
        "all_planned": _holm_by_family(pvalues, np.zeros(len(pairs))),
        "per_benchmark": _holm_by_family(pvalues, benchmarks),
        "executed_only": _holm_by_family(pvalues, np.where(ran, "executed", "unrun")),
    }
    z = float(norm.ppf(1 - alpha / 2))
    rows = []
    for index, row in enumerate(pairs):
        confirmation = row["confirmation"]
        result = {
            "benchmark": row["benchmark"],
            "model_a": row["model_a"],
            "model_b": row["model_b"],
            "status": row["status"],
            "pilot_gap": row["pilot"]["mean_difference"],
            "planned_n_items": row["planned_n_items"],
            "confirmation_gap": None,
            "pvalue": None,
            "detected": None,
            "same_sign_as_pilot": None,
            "holm_all_planned_p": float(adjusted["all_planned"][index]),
            "holm_per_benchmark_p": float(adjusted["per_benchmark"][index]),
            "holm_executed_only_p": None,
            "plug_in_normal_power": None,
            "plug_in_t_power": None,
        }
        if confirmation is not None:
            n = row["planned_n_items"]
            if confirmation["n_items"] != n:
                raise ValueError("confirmation item count differs from its plan")
            gap = row["pilot"]["mean_difference"]
            noncentrality = abs(gap) / math.sqrt(row["pilot"]["total_variance"] / n)
            critical = float(t.ppf(1 - alpha / 2, n - 1))
            result.update(
                confirmation_gap=confirmation["mean_difference"],
                pvalue=confirmation["pvalue"],
                detected=confirmation["detected"],
                same_sign_as_pilot=bool(np.sign(confirmation["mean_difference"]) == np.sign(gap)),
                holm_executed_only_p=float(adjusted["executed_only"][index]),
                plug_in_normal_power=_normal_power(noncentrality, z),
                plug_in_t_power=float(
                    nct.sf(critical, n - 1, noncentrality)
                    + nct.cdf(-critical, n - 1, noncentrality)
                ),
            )
        rows.append(result)
    tested = [row for row in rows if row["detected"] is not None]
    gaps = np.abs([row["confirmation_gap"] for row in tested])
    # Right-closed bins: a 1pp gap belongs to at_most_1pp, a 5pp gap to 1_to_5pp.
    bins = np.bincount(np.searchsorted([0.01, 0.05, 0.10, 0.20], gaps), minlength=5)
    above = [row["plug_in_t_power"] for row in tested if row["planned_n_items"] > floor]
    at_floor = [row["plug_in_t_power"] for row in tested if row["planned_n_items"] == floor]
    return {
        "evidence": "MEAS: committed fresh-item confirmations, not simulation",
        "detection_rule": "prespecified two-sided paired t-test on item means, p < alpha, "
        "no multiplicity correction",
        "alpha": alpha,
        "planned": len(rows),
        "tested": len(tested),
        "unrun": len(rows) - len(tested),
        "nominal_detections": sum(row["detected"] for row in tested),
        "detected_with_pilot_sign": sum(
            row["detected"] and row["same_sign_as_pilot"] for row in tested
        ),
        "holm_detections": {
            key: int(np.sum(values[ran] < alpha)) for key, values in adjusted.items()
        },
        "holm_families": {
            "all_planned": f"one family of all {len(rows)} planned pairs; unrun plans p=1",
            "per_benchmark": "one family of planned pairs per benchmark; unrun plans p=1",
            "executed_only": f"one family of the {len(tested)} executed confirmations",
        },
        "absolute_gap_quantiles": dict(
            zip(
                ("min", "q25", "median", "q75", "max"),
                (float(value) for value in np.quantile(gaps, [0, 0.25, 0.5, 0.75, 1])),
                strict=True,
            )
        ),
        "absolute_gap_bins": dict(
            zip(
                ("at_most_1pp", "1_to_5pp", "5_to_10pp", "10_to_20pp", "over_20pp"),
                (int(count) for count in bins),
                strict=True,
            )
        ),
        "minimum_confirm_items": floor,
        "comparisons_at_minimum": len(at_floor),
        "plug_in_t_power": {
            "min_above_minimum": min(above),
            "max_above_minimum": max(above),
            "min_at_minimum": min(at_floor),
            "sum_over_tested": math.fsum(row["plug_in_t_power"] for row in tested),
        },
        "pairs": rows,
    }
