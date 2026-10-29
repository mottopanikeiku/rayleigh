"""Shared-item CSV input: explicit IDs, strict binary scores, CLI agreement with .npy."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from eval_power import (
    discordance_upper_bound,
    exact_mcnemar_power,
    paired_variance,
    read_paired_csv,
    required_items,
)
from eval_power.cli import main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "gsm8k_qwen7_phi35_pilot.csv"


def write(tmp_path, text, encoding="utf-8"):
    path = tmp_path / "pilot.csv"
    path.write_text(text, encoding=encoding)
    return path


def test_committed_example_is_the_documented_pilot():
    ids, pilot = read_paired_csv(EXAMPLE)
    assert len(ids) == len(set(ids)) == 64
    assert pilot.dtype == np.uint8 and pilot.shape == (64, 2)
    assert int((pilot[:, 0] != pilot[:, 1]).sum()) == 10
    assert pilot[:, 0].mean() - pilot[:, 1].mean() == pytest.approx(-2 / 64)


def test_columns_are_matched_by_name_and_a_bom_is_accepted(tmp_path):
    ordered = read_paired_csv(write(tmp_path, "item_id,model_a,model_b\nq1,1,0\nq2,0,0\nq3,1,1\n"))
    shuffled = read_paired_csv(
        write(tmp_path, "\ufeffmodel_b,item_id,model_a\n0,q1,1\n0,q2,0\n1,q3,1\n", "utf-8")
    )
    for left, right in zip(ordered, shuffled, strict=True):
        np.testing.assert_array_equal(left, right)
    assert ordered[0].tolist() == ["q1", "q2", "q3"]
    assert ordered[1].tolist() == [[1, 0], [0, 0], [1, 1]]


@pytest.mark.parametrize(
    "text,message",
    [
        ("", "columns"),
        ("item_id,model_a\nq1,1\nq2,0\n", "columns"),
        ("item_id,model_a,model_b,extra\nq1,1,0,x\nq2,0,0,x\n", "columns"),
        ("item_id,model_a,model_b\nq1,1,0\nq1,0,0\n", "line 3"),
        ("item_id,model_a,model_b\nq1,1,0\n,0,0\n", "line 3"),
        ("item_id,model_a,model_b\nq1,1,0\n  ,0,0\n", "line 3"),
        ("item_id,model_a,model_b\nq1,1,0\nq2,0,0,1\n", "line 3"),
        ("item_id,model_a,model_b\nq1,1,0\nq2,0\n", "line 3"),
        ("item_id,model_a,model_b\nq1,2,0\nq2,0,0\n", "line 2"),
        ("item_id,model_a,model_b\nq1,1.0,0\nq2,0,0\n", "line 2"),
        ("item_id,model_a,model_b\nq1,true,0\nq2,0,0\n", "line 2"),
        ("item_id,model_a,model_b\nq1, 1,0\nq2,0,0\n", "line 2"),
        ("item_id,model_a,model_b\nq1,,0\nq2,0,0\n", "line 2"),
        ("item_id,model_a,model_b\nq1,1,0\n", "at least two"),
    ],
)
def test_malformed_tables_are_errors_not_silent_drops(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        read_paired_csv(write(tmp_path, text))


def test_cli_csv_matches_npy_and_reports_bound_plan(tmp_path, capsys):
    _, pilot = read_paired_csv(EXAMPLE)
    npy = tmp_path / "pilot.npy"
    np.save(npy, pilot)
    outputs = []
    for path in (EXAMPLE, npy):
        assert main(["--difference", "0.05", "--pilot", str(path)]) == 0
        outputs.append(json.loads(capsys.readouterr().out))
    assert outputs[0] == outputs[1]
    result = outputs[0]
    a, b = pilot[:, 0], pilot[:, 1]
    assert result["discordant_items"] == 10
    assert result["discordance_upper_95"] == discordance_upper_bound(10, 64)
    assert result["paired_items_per_model"] == required_items(0.05, paired_variance(a, b)) == 496
    assert result["bound_items_per_model"] == required_items(0.05, result["discordance_upper_95"])
    assert result["bound_items_per_model"] == 787
    assert any("Clopper-Pearson" in value for value in result["assumptions"])


def test_cli_zero_discordance_pilot_keeps_only_the_bound_plan(tmp_path, capsys):
    path = write(tmp_path, "item_id,model_a,model_b\n" + "".join(f"q{i},1,1\n" for i in range(32)))
    assert main(["--difference", "0.02", "--pilot", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["discordant_items"] == 0
    assert result["paired_items_per_model"] is None
    assert result["unpaired_items_per_model"] is None
    assert result["bound_items_per_model"] == required_items(0.02, 1 - 0.05 ** (1 / 32))


def test_cli_reports_csv_errors_without_traceback(tmp_path, capsys):
    path = write(tmp_path, "item_id,model_a,model_b\nq1,1,0\nq1,0,0\n")
    with pytest.raises(SystemExit) as exc:
        main(["--difference", "0.05", "--pilot", str(path)])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "line 3" in error and "Traceback" not in error


def test_example_script_output_is_the_one_in_the_readme(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["plan_from_pilot.py"])
    runpy.run_path(str(ROOT / "examples" / "plan_from_pilot.py"), run_name="__main__")
    output = capsys.readouterr().out
    assert "plug-in variance 496, discordance bound 787" in output
    assert f"{exact_mcnemar_power(0.05, 10 / 64, 496):.3f}" in output
    readme = (ROOT / "README.md").read_text()
    for line in output.splitlines():
        assert line in readme, line
