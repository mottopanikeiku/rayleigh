"""Independent finite-state references and nested-simulation contracts."""

import math
from itertools import product

import numpy as np
import pytest
from scipy.stats import binom

from eval_power import exact_mcnemar_power, holm_adjust, required_items
from eval_power.calibration import (
    FAMILY,
    RULES,
    clustered_rate,
    count_pvalues,
    family_adjust,
    nested_cell,
    paired_probabilities,
    plan_counts,
    shared_model_grid,
)


def enumerate_power(delta, q, n, alpha):
    """Enumerate win/loss/concordance cells, with integer null binomial sums."""
    total = 0.0
    for wins in range(n + 1):
        for losses in range(n - wins + 1):
            m = wins + losses
            pvalue = min(1, 2 * sum(math.comb(m, k) for k in range(min(wins, losses) + 1)) / 2**m)
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
    bounds = plan_counts(counts, 0.1, 0.4, "variance_only_bound")
    assert bounds[0] == bounds[1]  # exactly the same discordance, different signed gaps
    assert np.all(bounds >= plan_counts(counts, 0.1, 0.4, "fixed_effect"))


def test_no_discordance_does_not_mean_zero_budget():
    counts = np.array([[16, 0, 0, 16]])
    assert plan_counts(counts, 0.02, 0.05, "observed_gap")[0] == -1
    assert plan_counts(counts, 0.02, 0.05, "fixed_effect")[0] == -1
    assert plan_counts(counts, 0.02, 0.05, "variance_only_bound")[0] > 0


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
