"""Independent finite-state references and nested-simulation contracts."""

import math
from itertools import product

import numpy as np
import pytest
from scipy.stats import binom, nct, norm, t

from eval_power import discordance_upper_bound, exact_mcnemar_power, holm_adjust, required_items
from eval_power.calibration import (
    FAMILY,
    RULES,
    clustered_rate,
    confirmation_sensitivity,
    count_pvalues,
    exact_oracle_items,
    exact_wrong_sign,
    family_adjust,
    known_truth_budgets,
    nested_cell,
    paired_probabilities,
    plan_counts,
    shared_model_grid,
)


def enumerate_power(delta, q, n, alpha, wrong_sign=False):
    """Enumerate win/loss/concordance cells, with integer null binomial sums.

    With wrong_sign, count only rejections whose observed direction opposes delta.
    """
    total = 0.0
    for wins in range(n + 1):
        for losses in range(n - wins + 1):
            m = wins + losses
            pvalue = min(1, 2 * sum(math.comb(m, k) for k in range(min(wins, losses) + 1)) / 2**m)
            if wrong_sign and (wins - losses) * delta >= 0:
                continue
            if pvalue <= alpha:
                coefficient = math.comb(n, wins) * math.comb(n - wins, losses)
                total += (
                    coefficient
                    * ((q + delta) / 2) ** wins
                    * ((q - delta) / 2) ** losses
                    * (1 - q) ** (n - m)
                )
    return total


@pytest.mark.parametrize(
    "q,delta", [(0, 0), (0.01, 0.005), (0.2, 0), (0.4, -0.1), (0.8, 0.6), (1, 1)]
)
@pytest.mark.parametrize("n,alpha", [(1, 0.05), (6, 0.03125), (12, 0.125), (16, 0.05)])
def test_exact_power_matches_independent_cell_enumeration(q, delta, n, alpha):
    assert exact_mcnemar_power(delta, q, n, alpha) == pytest.approx(
        enumerate_power(delta, q, n, alpha), abs=2e-13
    )


def test_exact_power_all_discordances_in_one_direction():
    assert exact_mcnemar_power(1, 1, 5) == 0
    assert exact_mcnemar_power(1, 1, 6) == 1
    assert exact_mcnemar_power(0.01, 0.01, 128) == pytest.approx(binom.sf(5, 128, 0.01))


@pytest.mark.parametrize(
    "delta,q,n,alpha",
    [
        (0.2, 0.1, 16, 0.05),
        (0, -1, 16, 0.05),
        (0, 1.1, 16, 0.05),
        (float("nan"), 0.2, 16, 0.05),
        (0, float("nan"), 16, 0.05),
        (0, 0.2, 0, 0.05),
        (0, 0.2, True, 0.05),
        (0, 0.2, 1.5, 0.05),
        (0, 0.2, 16, 0),
        (0, 0.2, 16, 1),
        (0, 0.2, 16, float("nan")),
    ],
)
def test_exact_power_rejects_invalid_dgp(delta, q, n, alpha):
    with pytest.raises(ValueError):
        exact_mcnemar_power(delta, q, n, alpha)


def test_exact_null_is_conservative_and_sign_symmetric():
    for q, n in product((0.01, 0.1, 0.8, 1), (16, 64, 512)):
        assert exact_mcnemar_power(0, q, n) <= 0.05 + 1e-13
        assert exact_mcnemar_power(q / 2, q, n) == pytest.approx(exact_mcnemar_power(-q / 2, q, n))


def test_plans_match_public_normal_planner_and_ignore_gap_when_specified():
    counts = np.array([[10, 3, 7, 12], [10, 5, 5, 12]])
    for rule in ("observed_gap", "fixed_effect", "known_variance"):
        plans = plan_counts(counts, 0.1, 0.4, rule)
        for i, row in enumerate(counts):
            n = row.sum()
            gap = (row[2] - row[1]) / n
            variance = ((row[1] + row[2]) / n - gap**2) * n / (n - 1)
            if rule == "known_variance":
                variance = 0.4 - 0.1**2
            effect = gap if rule == "observed_gap" else 0.1
            assert plans[i] == (required_items(effect, variance) if effect else -1)
    bounds = plan_counts(counts, 0.1, 0.4, "discordance_bound")
    assert bounds[0] == bounds[1]  # exactly the same discordance, different signed gaps
    assert np.all(bounds >= plan_counts(counts, 0.1, 0.4, "fixed_effect"))


def test_no_discordance_does_not_mean_zero_budget():
    counts = np.array([[16, 0, 0, 16]])
    assert plan_counts(counts, 0.02, 0.05, "observed_gap")[0] == -1
    assert plan_counts(counts, 0.02, 0.05, "fixed_effect")[0] == -1
    assert plan_counts(counts, 0.02, 0.05, "discordance_bound")[0] > 0


def test_discordance_bound_is_one_sided_clopper_pearson():
    for d, n in ((0, 32), (3, 32), (10, 64), (127, 128)):
        bound = discordance_upper_bound(d, n)
        assert binom.cdf(d, n, bound) == pytest.approx(0.05, abs=1e-12)
    assert discordance_upper_bound(0, 32) == pytest.approx(1 - 0.05 ** (1 / 32))
    assert discordance_upper_bound(64, 64) == 1
    assert 10 / 64 < discordance_upper_bound(10, 64, 0.5) < discordance_upper_bound(10, 64)
    vector = discordance_upper_bound(np.array([0, 10, 64]), 64)
    assert vector.tolist() == [discordance_upper_bound(d, 64) for d in (0, 10, 64)]


@pytest.mark.parametrize(
    "d,n,confidence",
    [
        (1.0, 10, 0.95),
        (True, 10, 0.95),
        (-1, 10, 0.95),
        (11, 10, 0.95),
        (0, 0, 0.95),
        (1, 10, 0),
        (1, 10, 1),
        (1, 10, float("nan")),
    ],
)
def test_discordance_bound_rejects_invalid_counts(d, n, confidence):
    with pytest.raises(ValueError):
        discordance_upper_bound(d, n, confidence)


@pytest.mark.parametrize("delta,q", [(0.6, 0.8), (0.3, 0.5), (0.1, 0.5), (-0.2, 0.3)])
def test_exact_oracle_is_the_first_budget_reaching_target(delta, q):
    n = exact_oracle_items(delta, q)
    assert exact_mcnemar_power(delta, q, n) >= 0.8
    assert all(exact_mcnemar_power(delta, q, smaller) < 0.8 for smaller in range(1, n))
    if n <= 20:
        assert enumerate_power(delta, q, n, 0.05) >= 0.8 > enumerate_power(delta, q, n - 1, 0.05)


def test_gaussian_budget_underplans_small_exact_mcnemar_design():
    # Wald variance q-delta**2 omits the score test's null variance q, then discreteness.
    gaussian = required_items(0.6, 0.8 - 0.6**2)
    assert (gaussian, exact_oracle_items(0.6, 0.8)) == (10, 18)
    assert exact_mcnemar_power(0.6, 0.8, gaussian) < 0.5
    assert np.all(plan_counts(np.array([[1, 2, 3, 4]]), 0.6, 0.8, "exact_oracle") == 18)
    budgets = known_truth_budgets(0.6, 0.8)
    sizes = [budgets[f"{name}_n"] for name in ("gaussian_known_variance", "null_variance_gaussian")]
    assert sizes == [10, 15]  # ceil(((1.96*sqrt(.8) + .8416*sqrt(.44)) / .6)**2) = 15
    assert budgets["gaussian_known_variance_exact_power"] < 0.5
    assert 0.7 < budgets["null_variance_gaussian_exact_power"] < 0.8
    assert budgets["exact_oracle_exact_power"] >= 0.8


@pytest.mark.parametrize("delta,q,n", [(0.1, 0.4, 12), (-0.2, 0.5, 16), (0.005, 0.01, 6)])
def test_wrong_sign_rejection_matches_cell_enumeration(delta, q, n):
    expected = enumerate_power(delta, q, n, 0.05, wrong_sign=True)
    assert exact_wrong_sign(delta, q, n) == pytest.approx(expected, abs=2e-13)
    assert exact_wrong_sign(0, q, n) is None


def test_holm_preserves_full_family_and_matches_public_function():
    p = np.array([[[0.02, 0.03, 1], [0.001, 0.04, 0.04]], [[1, 1, 1], [0.05, 0.05, 0.05]]])
    expected = np.array([holm_adjust(row) for row in p.reshape(-1, FAMILY)]).reshape(p.shape)
    np.testing.assert_array_equal(family_adjust(p), expected)
    assert count_pvalues(np.array([0, 6]), np.array([0, 6])).tolist() == [1, 0.03125]


def test_mc_uncertainty_uses_outer_clusters_not_inner_trials():
    numerator = np.array([0.0, 0.5, 1.0, 0.5])
    result = clustered_rate(numerator)
    assert result["rate"] == 0.5
    assert result["mc_se"] == pytest.approx(numerator.std(ddof=1) / 2)
    conditional = clustered_rate(np.array([0.0, 0.5, 0.5, 0.0]), np.array([0.0, 1.0, 0.5, 0.5]))
    assert conditional["rate"] == 0.5
    assert clustered_rate(numerator * 0, numerator * 0)["rate"] is None


def test_nested_pilot_design_is_deterministic_and_accounts_for_every_plan():
    rows = nested_cell(0.005, 0.01, 0.005, 32, 20, 5, 20000, 731)
    assert rows == nested_cell(0.005, 0.01, 0.005, 32, 20, 5, 20000, 731)
    assert [r["rule"] for r in rows] == list(RULES)
    assert rows[0]["undefined_plans"] > 0
    for row in rows:
        assert row["undefined_plans"] + row["over_cap_plans"] + row["executed_plans"] == 60
        assert row["holm_workflow_rate"] <= row["nominal_workflow_rate"]
        if row["nominal_conditional_rate"] is not None:
            assert row["nominal_workflow_rate"] <= row["nominal_conditional_rate"]
    # A one-item cap cannot produce a rejection; no selection/drop of unrun plans.
    capped = nested_cell(0.02, 0.05, 0.02, 32, 10, 3, 1, 9)
    assert all(row["nominal_workflow_rate"] == 0 for row in capped)
    assert all(row["nominal_conditional_rate"] is None for row in capped)


def test_nested_oracle_reproduces_exact_power_and_reports_wrong_signs():
    rows = {r["rule"]: r for r in nested_cell(0.1, 0.5, 0.1, 32, 200, 16, 20000, 17)}
    oracle = rows["exact_oracle"]
    expected = exact_mcnemar_power(0.1, 0.5, exact_oracle_items(0.1, 0.5))
    assert abs(oracle["nominal_workflow_rate"] - expected) < 5 * oracle["nominal_workflow_mc_se"]
    for row in rows.values():
        assert 0 <= row["nominal_wrong_sign_rate"] <= row["nominal_workflow_rate"]
    weak = nested_cell(0.01, 0.5, 0.1, 32, 50, 8, 20000, 5)
    assert max(row["nominal_wrong_sign_rate"] for row in weak) > 0
    null = nested_cell(0, 0.5, 0.1, 32, 10, 2, 20000, 3)
    assert all(row["nominal_wrong_sign_rate"] is None for row in null)


def _pair(benchmark, pvalue, pilot_gap=0.2, gap=0.1, n=40):
    confirmation = None
    if pvalue is not None:
        confirmation = {
            "n_items": n,
            "mean_difference": gap,
            "pvalue": pvalue,
            "detected": pvalue < 0.05,
        }
    return {
        "benchmark": benchmark,
        "model_a": "a",
        "model_b": "b",
        "status": "feasible" if confirmation else "infeasible",
        "pilot": {"mean_difference": pilot_gap, "total_variance": 0.3},
        "planned_n_items": n if confirmation else None,
        "confirmation": confirmation,
    }


def test_confirmation_sensitivity_keeps_unrun_plans_in_planned_holm_families():
    summary = {
        "pairs": [
            _pair("x", 0.005, gap=0.01),
            _pair("x", 0.03, gap=-0.05, n=16),
            _pair("x", None),
            _pair("y", 0.02, gap=0.3),
            _pair("y", 0.2, gap=0.0101),
        ]
    }
    result = confirmation_sensitivity(summary, {"alpha": 0.05, "minimum_confirm_items": 16})
    assert (result["planned"], result["tested"], result["unrun"]) == (5, 4, 1)
    assert result["nominal_detections"] == 3
    assert result["detected_with_pilot_sign"] == 2
    # All five: .005*5, .02*4 -> 1. Executed four: .005*4, .02*3 -> 1.
    # Per benchmark: x {.005*3, .03*2, 1} -> 1; y {.02*2, .2} -> 1.
    assert result["holm_detections"] == {"all_planned": 1, "per_benchmark": 2, "executed_only": 1}
    assert result["absolute_gap_bins"] == {
        "at_most_1pp": 1,
        "1_to_5pp": 2,
        "5_to_10pp": 0,
        "10_to_20pp": 0,
        "over_20pp": 1,
    }
    assert result["comparisons_at_minimum"] == 1
    first = result["pairs"][0]
    noncentrality = 0.2 / math.sqrt(0.3 / 40)
    assert first["plug_in_normal_power"] == pytest.approx(
        norm.sf(norm.ppf(0.975) - noncentrality) + norm.cdf(-norm.ppf(0.975) - noncentrality)
    )
    critical = t.ppf(0.975, 39)
    assert first["plug_in_t_power"] == pytest.approx(
        nct.sf(critical, 39, noncentrality) + nct.cdf(-critical, 39, noncentrality)
    )
    assert first["plug_in_t_power"] < first["plug_in_normal_power"]
    assert result["pairs"][2]["holm_executed_only_p"] is None
    assert result["pairs"][2]["holm_all_planned_p"] == 1


def test_dgp_has_declared_truth():
    for q, delta in ((0.01, 0.005), (0.5, -0.1), (0.8, 0.6)):
        cells = paired_probabilities(delta, q)
        assert cells.sum() == pytest.approx(1)
        assert cells[1] + cells[2] == pytest.approx(q)
        assert cells[2] - cells[1] == pytest.approx(delta)


def test_shared_model_holm_controls_known_nulls_in_fixed_simulation():
    rows = shared_model_grid(531, 4000)
    for row in rows:
        if row["adjustment"] == "holm":
            assert row["fwer"] < 0.05 + 6 * math.sqrt(0.05 * 0.95 / row["trials"])
        assert 0 <= row["mc_low"] <= row["fwer"] <= row["mc_high"] <= 1
