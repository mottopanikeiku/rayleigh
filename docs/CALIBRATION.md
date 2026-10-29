# Known-truth planning calibration

This benchmark is **SIM: simulation**, not newly collected model evidence. It asks whether a nominal 80%-power budget delivers 80% rejection under the exact two-sided McNemar test when I know the truth. The repeated-decoding experiment uses paired t-tests on item means; this binary simulation does **not** calibrate that experiment's t-test. I report a separate plug-in t-power audit of those plans below, labelled as such.

## Reproduce

From a checkout with `uv sync --locked` completed:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run python scripts/calibrate.py
uv run python scripts/hero.py
uv run python scripts/check_results.py
```

The first command needs no downloads or model inference. It writes `results/known_truth/{configuration.json,reference.csv,nested.csv,shared_models.csv,confirmation_sensitivity.json}` and `figures/known_truth.svg`; the second draws the README hero from those files. The root seed is **20261029**, with NumPy PCG64 and the complete substream recipe in `configuration.json`. Defaults are 1,000 outer pilot families, 32 independent heldout families per pilot family, and 20,000 replicates per fixed-N reference or shared-model cell. Each nested family contains three independently generated model pairs, so each rule in each cell runs 96,000 tests. No timing claim is made.

`check_results.py`, which CI runs on x86-64 and arm64, regenerates every table and JSON document and compares them with the committed files (floats to a relative 1e-12, everything else exactly). It requires the regenerated calibration figure, both hero SVGs and the CSV example to match byte for byte. It then checks the claims the tables support: every null row at or below the 5% level within four Monte Carlo standard errors, Holm never above nominal, every plan accounted for, each Monte Carlo reference rate within 4.5 binomial standard errors of its exact value, the exact oracle's simulated rate within 4.5 standard errors of its exact power, and Holm familywise error at or below 5% in the shared-model panel.

## Known truth and reference

For binary outcomes A and B, let delta = E[A-B] and q = P[A != B]. The four cell probabilities, ordered 00, 01, 10, 11, are `(1-q)/2, (q-delta)/2, (q+delta)/2, (1-q)/2`. Thus Var(A-B) = q - delta². The grid includes q = 1%, 5%, 20%, 50%, 80%; meaningful effects are respectively 0.5, 2, 2, 10, 60 percentage points. Each is evaluated at true effects zero, half the meaningful effect, and the meaningful effect, with pilot sizes 32, 128, 512.

`exact_mcnemar_power(delta, discordance, n, alpha)` sums over **all** discordance counts M ~ Binomial(n,q). Conditional on M=m, the A-only count follows Binomial(m,(q+delta)/(2q)). The rejection region is exactly the two-sided conditional McNemar region, not a chi-squared or normal approximation, and a p-value equal to alpha rejects. The reference has no Monte Carlo error. Tests independently enumerate win/loss/concordance cells with integer binomial null probabilities, including equality at a discrete threshold, zero discordance, sign symmetry, and all-one-direction discordance.

`reference.csv` compares this reference with fixed-N Monte Carlo at N=16 to 8,192, the known-variance Gaussian budget and the exact oracle budget. Its Bonferroni column uses alpha/3; it is **not** exact Holm power. `exact_wrong_sign` is the exact probability of rejecting with the wrong sign (a type S error); it peaks at 0.79% (q=20%, half effect, N=64), where power is only 2.7%.

## Five planning rules

All rules plan for **nominal**, not familywise, 80% power at alpha=.05:

- **observed_gap:** Gaussian budget using the absolute pilot gap and unbiased sample variance. This is the noisy-gap diagnostic, not a recommended way to choose a meaningful effect.
- **fixed_effect:** prespecified meaningful effect, with unbiased sample Var(A-B) from the pilot. The observed gap enters only the variance calculation.
- **discordance_bound:** prespecified effect, with a one-sided 95% Clopper–Pearson upper bound on q as the planning variance (`discordance_upper_bound`). Since Var(A-B) <= q, this deliberately conservative rule uses only the pilot's discordance count, not its signed gap, and still plans when the pilot has no discordant items. It is Browne's upper-confidence-limit idea [1] applied to a binary paired variance. The bound does not make the Gaussian approximation exact and is not a familywise guarantee. The CLI reports it as `bound_items_per_model`.
- **known_variance:** oracle comparator using the true q - target² in the same Gaussian formula. Under a null or half-effect DGP it still plans for the meaningful effect, not for the realized truth.
- **exact_oracle:** the smallest N whose exact McNemar power at the true q and the meaningful effect reaches 80%. It isolates pilot noise and Gaussian approximation from the simulation machinery: its simulated rates (79.8–82.8%) reproduce its exact power (80.0–82.6%).

A zero-gap or zero-variance plug-in plan is undefined, not a zero-item study. Budgets above 20,000 are unrun, not truncated. Each result records undefined, over-cap and executed counts, budget quantiles among defined plans, and mean executed cost. Workflow rejection counts unrun plans as no rejection. Conditional rejection is restricted to executed plans and can look much better; both denominators are explicit. Unrun comparisons retain p=1 in the complete three-test Holm family. `nominal_wrong_sign_rate` counts rejections whose sign opposes the true effect; its largest value is 0.48% (q=20%, half effect, observed-gap rule, 32-item pilot), so two-sided rejection rates are not inflated by sign errors in this grid.

Within each outer replicate, a new pilot determines each budget, then 32 independent heldout families are drawn without reusing pilot observations. Monte Carlo SEs and approximate 95% intervals use the **outer replicate as the cluster**, including a ratio-of-means delta method for conditional rates. The inner trials are not treated as independent pilots. Zero-event normal intervals can collapse; fixed-N and shared-model rates use Wilson intervals instead. These are pointwise Monte Carlo intervals, not simultaneous guarantees across the grid. Rules share each cell's pilots but use separate heldout streams.

## What the recorded simulation says

At pilot N=128 and the meaningful effect, nominal workflow rejection fractions are:

| DGP | Observed gap | Fixed effect | Discordance bound | Known variance | Exact oracle |
|---|---:|---:|---:|---:|---:|
| q=1%, effect=0.5pp | 14.7% | 59.4% | 95.4% | 76.5% | 80.1% |
| q=5%, effect=2pp | 53.5% | 71.8% | 94.6% | 77.2% | 79.8% |
| q=20%, effect=2pp | 33.3% | 77.8% | 88.4% | 79.2% | 80.1% |
| q=50%, effect=10pp | 67.5% | 77.2% | 83.6% | 77.8% | 79.9% |
| q=80%, effect=60pp | 47.0% | 46.5% | 86.1% | 47.7% | 82.5% |

These percentages are SIM, from `nested.csv`; the complete file includes intervals, null controls, half effects, pilot sizes 32 and 512 and Holm sensitivity. For the q=20%, effect=2pp fixed-effect rule, the nominal estimate is **77.8% [77.4%, 78.2%]**, versus **72.4%** after Holm. For q=1%, the fixed-effect conditional rate is 80.4%, but **785/3,000 plans are undefined** because their pilots had no discordant items; its workflow rate is only 59.4%. The bound rule's median defined budget there is 11,467 versus 4,867 for the plug-in rule, and 129/3,000 bound plans exceed the cap, so the extra detections are not free. With a 32-item pilot at q=1%, every bound plan exceeds 20,000 items and the rule detects nothing: a pilot that small cannot plan a 0.5-point comparison.

Pilot size matters less than the rule. With 512-item pilots, observed-gap plans still detect only 45.6–74.6%; plug-in variance approaches the known-variance Gaussian rate (69.8–78.7%, 47.9% for q=80%); the bound delivers 81.3–95.5% at 1.1–2.1 times the plug-in median budget.

### Where the Gaussian budget loses power, even with the true variance

`configuration.json` records three known-q budgets per design and their exact power. The Gaussian formula uses q − δ² for both normal quantiles; the McNemar statistic's null variance is q. Using q for the alpha quantile and q − δ² for the power quantile gives the null-variance budget.

| DGP | Gaussian, true variance | Null-variance Gaussian | Exact oracle |
|---|---:|---:|---:|
| q=1%, effect=0.5pp | 3,132 items, 76.5% | 3,138, 76.6% | 3,373, 80.0% |
| q=5%, effect=2pp | 974, 77.3% | 979, 77.5% | 1,035, 80.0% |
| q=20%, effect=2pp | 3,917, 79.0% | 3,923, 79.1% | 4,016, 80.0% |
| q=50%, effect=10pp | 385, 77.6% | 391, 78.3% | 408, 80.0% |
| q=80%, effect=60pp | 10, 47.5% | 15, 72.9% | 18, 82.6% |

For large budgets the null variance barely matters and the exact test's discreteness and conservatism cost 1–3.5 points, recovered by 2.5–8% more items. For the 10-item design the variance mismatch accounts for most of the shortfall and discreteness for the rest. Exact power is sawtoothed in N, so a slightly larger N than the exact oracle can fall back below 80%; check neighbours with `exact_mcnemar_power` when a budget is small.

## Type-I error and dependent comparisons

Null rows in `reference.csv` and `nested.csv` report type-I behavior. Every rule stays at or below the nominal level: the largest null workflow rate is 4.69% [4.56%, 4.82%] (q=20%, fixed effect, 32-item pilot), because a plan never sees its heldout data. Three independent nominal tests reach a familywise rate of 13.4% there; Holm keeps every null family at or below 4.8%.

The nested comparisons are independent across pairs. A separate panel (`shared_models.csv`) tests all three pairs of three models on the **same items**: a latent easy/hard item has probability 1/2 each and model correctness probabilities .8/.2 conditional on difficulty. A mixed family offsets one model by .1, leaving one true-null pair. Joint eight-cell multinomial sampling preserves the shared-model/item dependence. At N=1,024 under the global null, the shared-model familywise false-positive fraction is **11.0% [10.6%, 11.4%] nominal** versus **3.77% [3.51%, 4.04%] Holm** (SIM). This is one controlled dependence structure at fixed sizes, not coverage of clustered prompts, repeated decodes, distribution shift, model selection, or adaptive stopping. IID items and a prespecified family are still assumptions.

## Measured confirmation sensitivity, kept separate

`confirmation_sensitivity.json` is **MEAS**, computed from the committed repeated-decoding `results/prospective/summary.json` and `protocol.json`, with both source hashes, and one row per planned pair. Detection is the protocol's prespecified rule: a two-sided paired t-test on item means, p < .05, no multiplicity correction. **23 of 27** executed confirmations are detected, all with the pilot's sign. Holm adjustment is a post hoc sensitivity analysis, so I report three families: **17** detections with all 30 planned pairs in one family and the three unrun plans as p=1, which is how the simulation treats unrun plans; 18 with the 27 executed tests alone; 19 with one 15-pair family per benchmark.

The realized absolute gaps are **0.15pp minimum, 11.3pp lower quartile, 16.9pp median, 40.0pp upper quartile, 57.5pp maximum**. One comparison is at most 1pp, two are 1–5pp, three 5–10pp, eight 10–20pp and thirteen above 20pp; nine use the protocol's 16-item minimum. This is mostly evidence about large separations, not prospective validation of hypothetical one-point budgets.

The protocol planned from each pilot's observed gap, the rule the simulation warns against for close models. As a check on the normal-versus-t mismatch, I recomputed each plan's power under its own pilot gap and planning variance with the t reference distribution (n−1 degrees of freedom): 76.4–79.8% for the 18 plans above the minimum and at least 83.0% for the nine at it, summing to 22.8 expected detections. These plug-in powers restate each plan's assumptions; they are not the comparisons' true power.

## References

1. Richard H. Browne. 1995. *On the use of a pilot sample for sample size determination*. Statistics in Medicine 14(17):1933–1940. DOI: 10.1002/sim.4780141709. [Opened PubMed abstract](https://pubmed.ncbi.nlm.nih.gov/8532986/): an upper one-sided confidence limit on the pilot standard deviation, rather than the pilot estimate, achieved the planned t-test power in most simulated trials.
2. Robert J. Connor. 1987. *Sample size for testing differences in proportions for the paired-sample design*. Biometrics 43(1):207–211. [Opened PubMed abstract](https://pubmed.ncbi.nlm.nih.gov/3567305): Miettinen's matched-pair approximation tends to overestimate power relative to Duffy's exact power, and Connor gives a more conservative approximation. I did not open the full text, so the null-variance budget above is my derivation, not a reproduction of Connor's formula.
