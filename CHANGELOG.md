# Changelog

All notable changes to this project are listed here. Versions follow [semantic versioning](https://semver.org/); the version in `pyproject.toml` and `CITATION.cff` is the one under preparation. [Release steps](docs/RELEASING.md) describe how a version is tagged.

## 0.2.0 (unreleased)

### Added

- A planning simulation (`scripts/calibrate.py`, `src/eval_power/calibration.py`, [methods](docs/CALIBRATION.md)): five paired designs, three true effects, three pilot sizes and five planning rules, tested with exact McNemar tests, Holm correction and Monte Carlo intervals clustered by pilot.
- An exact-power comparison showing where Gaussian budgets fall short even with the true variance.
- Adjusted tests for the repeated-decoding experiment, with effect sizes, sign agreement and t-test power under each plan's assumptions.
- `exact_mcnemar_power`, the exact power of the two-sided conditional McNemar test, and `discordance_upper_bound`, a one-sided Clopper–Pearson bound on the discordance rate.
- Shared-item CSV input (`read_paired_csv` and `--pilot table.csv`), with explicit item IDs and errors that name the offending line.
- `bound_items_per_model` and `discordant_items` in the CLI pilot output.
- The `rayleigh` command, a GSM8K pilot CSV, and a runnable API example (`examples/`).
- `CITATION.cff`, this changelog and release steps.
- CI recomputes the simulation tables, figures, README image and example CSV; it also installs the wheel and runs the CLI example.

### Changed

- Shorter README, calculator copy and methods, with practical planning advice and a Python example.
- The README image is drawn by `scripts/hero.py` from the result files.
- A CLI pilot with zero paired or unpaired variance now reports that budget as null instead of exiting with an error, because the discordance bound still gives a finite plan.
- Cache cleanup prints the removed models and byte count without writing a result file.

## 0.1.0 (untagged)

- Gaussian item-count planning for paired binary outcomes, with the browser calculator.
- Pilot-planning audits on five pinned public correctness matrices.
- The prospective repeated-decoding experiment on six open models, and the historical Arena vote analysis.
