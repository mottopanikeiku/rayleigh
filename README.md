# rayleigh

How many benchmark items does it take to tell two LLMs apart? [Use the paired sample-size calculator](https://mottopanikeiku.github.io/rayleigh/), or plan from your own pilot with Python.

I tested five planning rules in simulations with an 80% power target. With a 128-item pilot, choosing the budget from the observed accuracy gap detected a difference only **15–68% of the time**. Fixing a meaningful gap first and using a 95% upper bound on the discordance rate reached **84–95%**, at **1.2–2.4×** the median budget from the pilot variance.

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/hero-dark.svg"><img alt="Detection rates for five sample-size rules in paired-evaluation simulations" src="docs/hero-light.svg" width="760"></picture>

## Choose the gap before the pilot

Use the smallest accuracy difference you care about. Let the pilot estimate how often the models disagree: an item is discordant when exactly one model gets it right.

| Situation | Planning rule | Simulation result at a 128-item pilot |
|---|---|---|
| Disagreement on at least 20% of items, with budgets of hundreds or more | Pilot variance, with the target gap fixed | 77–78% power, slightly below the 80% target |
| Disagreement on 5% of items or fewer | 95% upper bound on discordance | About 95% power, versus 59–72% with pilot variance |
| Very small budgets | Check exact McNemar power | One Gaussian plan needed 10 items and had 48% power; the exact calculation needed 18 |

The bound buys more room for pilot error. It can also make a study unaffordable: at 1% discordance, every bound plan from a 32-item pilot exceeded the 20,000-item limit. [Simulation methods and full tables](docs/CALIBRATION.md).

## Plan from your own scores

The CSV needs one row per shared item and the columns `item_id`, `model_a`, `model_b`. Scores must be 0 or 1, and item IDs must be unique. The example contains 64 GSM8K questions answered greedily by Qwen2.5-7B and Phi-3.5-mini.

```bash
uv sync --locked
uv run rayleigh --difference 0.05 --pilot examples/gsm8k_qwen7_phi35_pilot.csv
```

`--difference 0.05` means a five-percentage-point gap. The JSON output includes:

```text
"pilot_n": 64,
"discordant_items": 10,
"paired_items_per_model": 496,
"bound_items_per_model": 787,
"unpaired_items_per_model": 611
```

Use the same planner from Python:

```python
from eval_power import discordance_upper_bound, read_paired_csv, required_items

item_ids, scores = read_paired_csv("examples/gsm8k_qwen7_phi35_pilot.csv")
discordant = int((scores[:, 0] != scores[:, 1]).sum())
variance_bound = discordance_upper_bound(discordant, len(item_ids))
n = required_items(0.05, variance_bound)  # 787 shared items
```

[The longer example](examples/plan_from_pilot.py) compares both budgets and checks their exact power at the pilot's estimated discordance:

```text
$ uv run python examples/plan_from_pilot.py
pilot: 64 shared items, 10 discordant, gap -0.031
items per model: plug-in variance 496, discordance bound 787
exact power at q=0.156, 5-point gap, plug-in budget of 496: 0.781
exact power at q=0.156, 5-point gap, bound budget of 787: 0.939
```

The Python import is `eval_power`; the command is `rayleigh`. Install from this checkout: the PyPI package named `rayleigh` is unrelated. The browser calculator uses the Gaussian planner with a supplied variance. The Python API and CLI add the pilot-based discordance bound; `exact_mcnemar_power` is available in the Python API.

## How the simulation works

For each design, I set the true accuracy gap and the fraction of discordant items. Each pilot chooses a budget, then a separate sample of binary outcomes gets an exact two-sided McNemar test at alpha 0.05. Five designs cover discordance from 1% to 80%, with pilot sizes of 32, 128 and 512.

Each rule runs 96,000 tests per design, effect and pilot size: 1,000 pilot families, three comparisons per family, and 32 evaluation samples per pilot. Undefined plans and budgets above 20,000 items count as failures to detect a difference. The simulation also checks zero effects, wrong-sign detections and Holm correction across three comparisons.

The [full results](results/known_truth/nested.csv) include detection rates, item counts and Monte Carlo intervals. [Prior work](docs/PRIOR_WORK.md) covers paired power analysis and the older statistical problem of planning from a noisy pilot effect.

## Tests on six small models

I generated 13,830 answers from Qwen2.5 1.5B, 3B and 7B, SmolLM2 1.7B, Phi-3.5-mini and Mistral-7B-v0.3 on GSM8K and ARC-Challenge. Each model answered 64 pilot questions per benchmark once greedily and five times at temperature 0.7. The pilots set the budgets for comparisons on new questions.

**23 of 27 comparisons detected a difference** with paired t-tests on five-answer item means. That falls to **17** after Holm correction across all 30 planned comparisons, including three above the collection limit. The median measured gap was **16.9 percentage points**. [Per-pair results](results/prospective/summary.json) and [effect sizes and adjusted tests](results/known_truth/confirmation_sensitivity.json).

![Pilot variance and paired accuracy differences on new questions](figures/prospective.svg)

The repository also includes an analysis of 54,985 historical Arena votes. Only 4 of 54 adjacent model pairs had 95% intervals excluding zero. [Preference-vote methods and results](docs/PRIOR_WORK.md#preference-votes).

## Run the checks

```bash
uv run pytest -q tests site/test_fixture.py
uv run python site/export_fixture.py --check
node --test site/*.test.mjs
node site/build.mjs
```

CI also runs `scripts/check_results.py`, which recomputes the simulation tables and checks saved responses, figures and examples. [Release notes](CHANGELOG.md) · [Citation](CITATION.cff).

## Limitations

- The simulation assumes independent binary items. Clustered questions, prompt changes, repeated decoding and adaptive stopping need their own power analysis.
- The bound targets one comparison at a time. Plan for the full family when testing several pairs; multiplicity correction reduces power.
- The six-model experiment mostly tested large gaps. It used observed-gap planning and paired t-tests, so its detections do not validate one-point budgets or the discordance-bound rule.
- Public benchmarks may contain training data. ARC uses guided answer-choice generation; GSM8K uses final-number extraction. The [collection details](docs/PRIOR_WORK.md#repeated-answers) explain the grading and seed fixes made before confirmation.
