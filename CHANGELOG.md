# Changelog

All notable changes to this project are listed here. Versions follow [semantic versioning](https://semver.org/); the version in `pyproject.toml` and `CITATION.cff` is the one under preparation. [Release steps](docs/RELEASING.md) describe how a version is tagged.

## 0.2.0 (unreleased)

### Added

- A known-truth planning calibration (`scripts/calibrate.py`, `src/eval_power/calibration.py`, [methods](docs/CALIBRATION.md)): five paired designs, three true effects, three pilot sizes and five planning rules, judged by the exact McNemar test, with null controls, Holm families, type S errors, a shared-model dependence panel and Monte Carlo errors clustered by pilot.
- An exact-oracle comparator and a record of where Gaussian budgets lose power even with the true variance.
- A measured sensitivity audit of the repeated-decoding confirmations: Holm counts for three family definitions, the realized effect-size distribution, sign agreement and plug-in t-test power of each plan.
- `exact_mcnemar_power`, the exact power of the two-sided conditional McNemar test, and `discordance_upper_bound`, a one-sided Clopper–Pearson bound on the discordance rate.
- Shared-item CSV input (`read_paired_csv` and `--pilot table.csv`), with explicit item IDs and errors that name the offending line.
- `bound_items_per_model` and `discordant_items` in the CLI pilot output.
- The `rayleigh` command, a committed GSM8K pilot CSV, and a runnable API example (`examples/`).
- `CITATION.cff`, this changelog and release steps.
- CI regenerates every known-truth table, figure, the README hero and the CSV example and requires them to match; it also installs the built wheel in a clean environment and runs the CLI example.

### Changed

- The README leads with which planning rule to trust and when; the measured headline now sits beside its Holm counts and effect-size distribution.
- The README hero is drawn by `scripts/hero.py` from committed results instead of being a hand-made image.
- A CLI pilot with zero paired or unpaired variance now reports that budget as null instead of exiting with an error, because the discordance bound still gives a finite plan.

## 0.1.0 (untagged)

- Gaussian item-count planning for paired binary outcomes, with the browser calculator.
- Pilot-planning audits on five pinned public correctness matrices.
- The prospective repeated-decoding experiment on six open models, and the historical Arena vote analysis.
