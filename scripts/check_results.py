"""CI checks for matrix provenance, complete hypothesis families, and result consistency."""

import csv
import gzip
import hashlib
import json
import math
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from calibrate import build, make_figure
from export_example import EXAMPLE, build_example
from hero import THEMES, make_hero
from pilot_revision import build_revision

from eval_power.analysis import BENCHMARKS
from eval_power.arena import ELO_SCALE, contrast, fit_bt, planned_votes
from eval_power.calibration import ALPHA, POWER, exact_oracle_items
from eval_power.grading import grade_answer, grade_strict_answer
from eval_power.prospective import analyze_plan, analyze_strict_secondary, validate_sources
from eval_power.stats import exact_mcnemar_power, holm_adjust


def rows(name: str) -> list[dict]:
    path = Path("results") / name
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_matches(actual, expected, path: str = "") -> None:
    """Require identical JSON structure and values, allowing last-bit float differences.

    SciPy's distribution functions can differ in the final bits across CPU architectures,
    so floats use a 1e-12 relative tolerance; everything else must match exactly.
    """
    if isinstance(expected, float) or isinstance(actual, float):
        assert isinstance(actual, float) and isinstance(expected, float), path
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), (
            f"{path}: {actual} vs {expected}"
        )
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            assert_matches(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_matches(left, right, f"{path}[{index}]")
    else:
        assert type(actual) is type(expected) and actual == expected, f"{path}: {actual}"


def check_prospective() -> None:
    directory = Path("results/prospective")
    raw_directory = directory / "primary"
    protocol_path, plan_path = directory / "protocol.json", raw_directory / "plan.json"
    protocol = json.loads(protocol_path.read_text())
    plan = json.loads(plan_path.read_text())
    summary = json.loads((directory / "summary.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    commit = provenance["plan_commit"]
    assert len(commit) == 40 and all(char in "0123456789abcdef" for char in commit)
    for path, key in ((plan_path, "plan_sha256"), (protocol_path, "protocol_sha256")):
        committed = subprocess.check_output(["git", "show", f"{commit}:{path}"])
        assert committed == path.read_bytes()
        assert provenance[key] == hashlib.sha256(committed).hexdigest()
    committed_time = datetime.fromisoformat(
        subprocess.check_output(["git", "show", "-s", "--format=%cI", commit], text=True).strip()
    ).astimezone(UTC)
    assert committed_time == datetime.fromisoformat(provenance["plan_commit_utc"])
    assert provenance["push_completed_before_collection"] is True
    push_time = datetime.fromisoformat(provenance["push_recorded_utc"])
    stages = {}
    item_references = {}
    for stage in ("pilot", "confirm"):
        documents = {}
        for path in sorted((raw_directory / stage).glob("*.json.gz")):
            with gzip.open(path, "rt") as stream:
                document = json.load(stream)
            slug = path.name.removesuffix(".json.gz")
            assert document["slug"] == slug
            if stage == "confirm":
                assert datetime.fromisoformat(document["metadata"]["started_utc"]) > push_time
            documents[slug] = document
            for benchmark, items in document["benchmarks"].items():
                for item in items:
                    identity = (benchmark, item["item_id"])
                    reference = (item["prompt_sha256"], item["reference"])
                    assert item_references.setdefault(identity, reference) == reference
                    for answer in (item["greedy"], *item["samples"]):
                        prediction, correct = grade_answer(
                            benchmark, answer["text"], item["reference"]
                        )
                        assert answer["prediction"] == prediction
                        assert answer["correct"] == correct
                        strict_prediction, strict_correct = grade_strict_answer(
                            benchmark, answer["text"], item["reference"]
                        )
                        assert answer["strict_prediction"] == strict_prediction
                        assert answer["strict_correct"] == strict_correct
                        if benchmark == "arc":
                            assert answer["text"] == f"Answer: {prediction}"
                            assert correct == strict_correct
        stages[stage] = documents
    validate_sources(
        plan,
        protocol,
        stages["pilot"],
        stages["confirm"],
        hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    recalculated = json.loads(
        json.dumps(analyze_plan(plan, stages["pilot"], stages["confirm"]), allow_nan=False)
    )
    for key, value in recalculated.items():
        assert_matches(value, summary[key], f"prospective summary {key}")
    strict_secondary = analyze_strict_secondary(plan, stages["pilot"], stages["confirm"])
    assert_matches(
        json.loads(json.dumps(strict_secondary, allow_nan=False)),
        summary["strict_secondary"],
        "prospective summary strict_secondary",
    )
    for group in ("protocol", "plan"):
        source = summary["sources"][group]
        assert source["sha256"] == hashlib.sha256(Path(source["file"]).read_bytes()).hexdigest()
    for stage in ("pilot", "confirm"):
        for source in summary["sources"][stage]:
            assert source["sha256"] == hashlib.sha256(Path(source["file"]).read_bytes()).hexdigest()
    assert ET.parse("figures/prospective.svg").getroot().tag.endswith("svg")
    assert_matches(
        build_revision(),
        json.loads((directory / "pilot_revision.json").read_text()),
        "pilot_revision.json",
    )


def assert_csv_matches(path: Path, expected: list[dict]) -> None:
    """Committed CSV text versus regenerated rows, with assert_matches float tolerance."""
    with path.open(newline="", encoding="utf-8") as handle:
        committed = list(csv.DictReader(handle))
    assert len(committed) == len(expected), f"{path}: row count"
    for index, (row, reference) in enumerate(zip(committed, expected, strict=True)):
        assert list(row) == list(reference), f"{path}[{index}]: columns"
        for key, value in reference.items():
            where = f"{path}[{index}].{key}"
            if value is None:
                assert row[key] == "", where
            elif isinstance(value, float):
                assert_matches(float(row[key]), value, where)
            else:
                assert row[key] == str(value), where


def check_known_truth() -> None:
    """Regenerate every known-truth artifact, compare, then test the claims it supports."""
    directory = Path("results/known_truth")
    configuration = json.loads((directory / "configuration.json").read_text())
    result, sensitivity = build(
        configuration["seed"],
        configuration["outer"],
        configuration["inner"],
        configuration["reference_trials"],
        configuration["cap"],
    )
    for key in ("reference", "nested", "shared_models"):
        assert_csv_matches(directory / f"{key}.csv", result[key])
    for key, document in (
        ("configuration", result["configuration"]),
        ("confirmation_sensitivity", sensitivity),
    ):
        assert_matches(
            json.loads(json.dumps(document, allow_nan=False)),
            json.loads((directory / f"{key}.json").read_text()),
            f"known_truth {key}",
        )
    with tempfile.TemporaryDirectory() as scratch:
        figures = [(Path(scratch) / "known_truth.svg", Path("figures/known_truth.svg"))]
        make_figure(result["nested"], figures[0][0])
        for theme in THEMES:
            figures.append((Path(scratch) / f"hero-{theme}.svg", Path(f"docs/hero-{theme}.svg")))
            make_hero(theme, figures[-1][0])
        for fresh, committed in figures:
            assert fresh.read_bytes() == committed.read_bytes(), f"{committed} is stale"
    for row in result["reference"]:
        exact, rate = row["exact_power"], row["mc_rate"]
        if exact in (0, 1):
            assert rate == exact, row
        else:
            assert abs(rate - exact) <= 4.5 * math.sqrt(exact * (1 - exact) / row["trials"]), row
    for row in result["nested"]:
        accounted = row["undefined_plans"] + row["over_cap_plans"] + row["executed_plans"]
        assert accounted == row["total_plans"], row
        assert row["holm_workflow_rate"] <= row["nominal_workflow_rate"], row
        if row["delta"] == 0:
            # Plans never see heldout data, so the exact test keeps its level.
            assert row["nominal_workflow_rate"] <= ALPHA + 4 * row["nominal_workflow_mc_se"], row
        if row["rule"] == "exact_oracle" and row["delta"] == row["target"]:
            n = exact_oracle_items(row["target"], row["discordance"])
            expected = exact_mcnemar_power(row["delta"], row["discordance"], n)
            assert expected >= POWER
            assert (
                abs(row["nominal_workflow_rate"] - expected) <= 4.5 * row["nominal_workflow_mc_se"]
            ), row
    for row in result["shared_models"]:
        if row["adjustment"] == "holm":
            assert row["fwer"] <= ALPHA + 4 * math.sqrt(ALPHA * (1 - ALPHA) / row["trials"]), row
    assert EXAMPLE.read_text(encoding="utf-8") == build_example(), f"{EXAMPLE} is stale"


def check_arena() -> None:
    manifest = json.loads(Path("data/arena_manifest.json").read_text())
    directory = Path("results/arena")
    aggregate = Path(manifest["aggregate_file"])
    assert hashlib.sha256(aggregate.read_bytes()).hexdigest() == manifest["aggregate_sha256"]
    summary = json.loads((directory / "summary.json").read_text())
    with np.load(aggregate, allow_pickle=False) as data:
        names = data["models"].tolist()
        pilot, heldout = data["pilot"], data["heldout"]
    full = pilot + heldout
    assert len(names) == manifest["models"]
    assert int(full.sum()) == manifest["source_votes"]
    assert int(pilot.sum()) == manifest["pilot_votes"]
    assert int(heldout.sum()) == manifest["heldout_votes"]
    a, b = np.triu_indices(len(names), 1)
    degree = np.bincount(a, full.sum(axis=1), minlength=len(names))
    degree += np.bincount(b, full.sum(axis=1), minlength=len(names))
    eligible = np.flatnonzero(degree >= summary["minimum_model_exposure"])
    mask = np.isin(a, eligible) & np.isin(b, eligible)
    names = [names[i] for i in eligible]
    full, pilot, heldout = full[mask], pilot[mask], heldout[mask]
    assert len(names) == summary["eligible_models"]
    assert int(full.sum()) == summary["retained_votes"]
    assert int(pilot.sum()) == summary["pilot_votes"]
    assert int(heldout.sum()) == summary["heldout_votes"]
    fit, pilot_fit, heldout_fit = (fit_bt(counts, len(names)) for counts in (full, pilot, heldout))
    assert fit.status == summary["fit_status"] == "ok"
    assert pilot_fit.status == summary["pilot_fit_status"]
    assert heldout_fit.status == summary["heldout_fit_status"]
    ranking = np.argsort(-fit.scores, kind="stable")
    ranked = rows("arena/ranking.csv")
    assert [row["model"] for row in ranked] == [names[i] for i in ranking]
    np.testing.assert_allclose(
        [float(row["score_log_odds"]) for row in ranked], fit.scores[ranking], atol=1e-9
    )
    with np.load(directory / "bootstrap.npz", allow_pickle=False) as data:
        boots = data["scores"]
    assert boots.shape == (summary["bootstrap_successes"], len(names))
    assert np.all(np.isfinite(boots))
    np.testing.assert_allclose(boots.sum(axis=1), 0, atol=1e-8)
    assert sum(summary["bootstrap_statuses"].values()) == summary["bootstrap_attempts"]
    adjacent = rows("arena/adjacent.csv")
    assert len(adjacent) == len(names) - 1 == summary["adjacent_pairs"]
    budgets, wald_count, bootstrap_count = [], 0, 0
    for row, i, j in zip(adjacent, ranking[:-1], ranking[1:], strict=True):
        assert (row["model_a"], row["model_b"]) == (names[i], names[j])
        gap, se = contrast(fit, i, j)
        wald = np.array([gap - 1.96 * se, gap + 1.96 * se]) * ELO_SCALE
        bootstrap = np.quantile(boots[:, i] - boots[:, j], [0.025, 0.975]) * ELO_SCALE
        np.testing.assert_allclose(
            [float(row["wald_low_elo"]), float(row["wald_high_elo"])], wald, atol=1e-8
        )
        np.testing.assert_allclose(
            [float(row["bootstrap_low_elo"]), float(row["bootstrap_high_elo"])],
            bootstrap,
            atol=1e-8,
        )
        budget, status = planned_votes(gap, se, int(full.sum()))
        assert row["plan_status"] == status
        assert int(row["total_votes_for_80pct"]) == budget
        budgets.append(budget)
        wald_count += int(wald[0] > 0 or wald[1] < 0)
        bootstrap_count += int(bootstrap[0] > 0 or bootstrap[1] < 0)
    assert wald_count == summary["wald_intervals_excluding_zero"]
    assert bootstrap_count == summary["bootstrap_intervals_excluding_zero"]
    assert np.median(budgets) == summary["median_adjacent_total_votes_for_80pct"]
    planning = rows("arena/pilot_planning.csv")
    starts = np.unique(np.linspace(0, len(names) - 2, len(planning), dtype=int))
    assert len(planning) == summary["score_independent_planned_pairs"]
    for row, i in zip(planning, starts, strict=True):
        assert (row["model_a"], row["model_b"]) == (names[i], names[i + 1])
        gap, se = contrast(pilot_fit, i, i + 1)
        budget, status = planned_votes(gap, se, int(pilot.sum()))
        assert int(row["planned_total_votes"]) == budget and row["plan_status"] == status
        trials, rejections = int(row["trials"]), int(row["rejections"])
        rate = float(row["rejection_rate_all_trials"])
        assert 0 <= rejections <= trials
        assert abs(rate * trials - rejections) < 1e-10
        assert (
            sum(
                int(row[key])
                for key in ("successful_fits", "disconnected", "separated", "other_fit_failures")
            )
            == trials
        )
    np.testing.assert_allclose(
        np.median([float(row["rejection_rate_all_trials"]) for row in planning]),
        summary["median_heldout_resampling_rejection_rate"],
    )
    assert ET.parse("figures/arena_votes.svg").getroot().tag.endswith("svg")


def main() -> None:
    check_arena()
    manifest = json.loads(Path("data/manifest.json").read_text())
    configuration = json.loads(Path("results/configuration.json").read_text())
    audit = {r["benchmark"]: r for r in rows("audit.csv")}
    adjacency = rows("adjacent.csv.gz")
    with np.load("results/all_pair_tests.npz", allow_pickle=False) as tests:
        for benchmark in BENCHMARKS:
            output = manifest["outputs"][benchmark]
            path = Path("data") / output["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == output["sha256"]
            with np.load(path, allow_pickle=False) as source:
                y, models, subjects = source["correctness"], source["models"], source["subjects"]
                assert list(y.shape) == output["shape"]
                assert y.dtype == np.uint8 and np.all((y == 0) | (y == 1))
                assert len(models) == len(set(models)) == 395
                assert models.tolist() == manifest["source"]["model_ids"]
                assert len(source["item_ids"]) == len(set(source["item_ids"])) == len(y)
                assert len(subjects) == len(y)
                assert len(np.unique(subjects)) == (57 if benchmark == "mmlu" else len(y))
                ordering = np.lexsort((models, -y.mean(axis=0)))
            i, j = tests[f"{benchmark}_i"], tests[f"{benchmark}_j"]
            expected = np.triu_indices(395, 1)
            np.testing.assert_array_equal(i, expected[0])
            np.testing.assert_array_equal(j, expected[1])
            np.testing.assert_allclose(
                tests[f"{benchmark}_holm"], holm_adjust(tests[f"{benchmark}_p"]), rtol=1e-13
            )
            a = audit[benchmark]
            assert int(a["all_pair_family"]) == 395 * 394 // 2
            assert int(a["items"]) == output["shape"][0]
            sub = [r for r in adjacency if r["benchmark"] == benchmark]
            assert len(sub) == 394
            assert [r["model_a"] for r in sub] == models[ordering[:-1]].tolist()
            assert [r["model_b"] for r in sub] == models[ordering[1:]].tolist()
            significant = sum(float(r["full_family_holm_p"]) <= 0.05 for r in sub)
            assert significant == int(a["paired_full_holm_distinguishable"])
            assert (394 - significant) / 394 == float(
                a["paired_full_holm_not_distinguishable_fraction"]
            )
            if benchmark == "mmlu":
                np.testing.assert_allclose(
                    tests["mmlu_cluster_holm"], holm_adjust(tests["mmlu_cluster_p"]), rtol=1e-13
                )
    seen = set()
    for row in rows("validation_curves.csv.gz"):
        key = tuple(
            row[field] for field in ("benchmark", "pilot_n", "split", "pair", "design", "n")
        )
        assert key not in seen
        seen.add(key)
        assert int(row["trials"]) == configuration["trials_per_curve_point"]
        assert int(row["n"]) <= int(row["heldout_n"])
        rate = float(row["observed_paired"])
        assert 0 <= float(row["mc_low"]) <= rate + 1e-14
        assert rate - 1e-14 <= float(row["mc_high"]) <= 1
        assert abs(rate * int(row["trials"]) - round(rate * int(row["trials"]))) < 1e-9
    for name in (
        "pilot_planning.svg",
        "leaderboard_noise.svg",
        "minimum_difference.svg",
        "pilot_calibration.svg",
    ):
        assert ET.parse(Path("figures") / name).getroot().tag.endswith("svg")
    assert Path("README.md").read_text().rstrip().endswith("Written with AI coding assistance.")
    check_prospective()
    check_known_truth()
    print(
        "Validated five pinned binary matrices, all-pair Holm families, "
        "calibration rows, fresh-item response counts and grades, and SVGs. "
        "Regenerated the known-truth tables, sensitivity audit, calibration figure, "
        "README hero and CSV example; all match."
    )


if __name__ == "__main__":
    main()
