# Known-truth planning calibration

This benchmark is **SIM: simulation**, not newly collected model evidence. It asks whether a nominal 80%-power Gaussian budget delivers 80% rejection under the exact two-sided McNemar test. The old repeated-decoding experiment uses paired t-tests on item means; this binary simulation does **not** calibrate that experiment's small-sample t-test.

## Reproduce

From a checkout with `uv sync --locked` completed:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run python scripts/calibrate.py
```

This command needs no downloads or model inference. It writes `results/known_truth/{configuration.json,reference.csv,nested.csv,shared_models.csv,confirmation_sensitivity.json}` and `figures/known_truth.svg`. The root seed is **20261029**, with NumPy PCG64 and the complete substream recipe in `configuration.json`. Defaults are 1,000 outer pilot families, 32 independent heldout families per pilot family, and 20,000 replicates per fixed-N reference or shared-model cell. Each nested family contains three independently generated model pairs. No timing claim is made.

## Known truth and reference

For binary outcomes A and B, let delta = E[A-B] and q = P[A != B]. The four cell probabilities, ordered 00, 01, 10, 11, are `(1-q)/2, (q-delta)/2, (q+delta)/2, (1-q)/2`. Thus Var(A-B) = q - delta². The grid includes q = 1%, 5%, 20%, 50%, 80%; meaningful effects are respectively 0.5, 2, 2, 10, 60 percentage points. Each is evaluated at true effects zero, half the meaningful effect, and the meaningful effect, with pilot sizes 32, 128, 512.

`exact_mcnemar_power(delta, discordance, n, alpha)` sums over **all** discordance counts M ~ Binomial(n,q). Conditional on M=m, the A-only count follows Binomial(m,(q+delta)/(2q)). The rejection region is exactly the two-sided conditional McNemar region, not a chi-squared or normal approximation. The reference has no Monte Carlo error. Tests independently enumerate win/loss/concordance cells with integer binomial null probabilities, including equality at a discrete threshold, zero discordance, sign symmetry, and all-one-direction discordance.

`reference.csv` compares this reference with fixed-N Monte Carlo, including N=16 and the known-variance Gaussian budget. Its Bonferroni column uses alpha/3; it is **not** exact Holm power.

## Four planning rules

All rules plan for **nominal**, not familywise, 80% power at alpha=.05:

- **observed_gap:** Gaussian budget using the absolute pilot gap and unbiased sample variance. This is the noisy-gap diagnostic, not a recommended way to choose a meaningful effect.
- **fixed_effect:** prespecified meaningful effect, with unbiased sample Var(A-B) from the pilot. The observed gap enters only the variance calculation.
- **variance_only_bound:** prespecified effect, with a one-sided 95% Clopper–Pearson upper bound on q as the planning variance. Since Var(A-B) <= q, this deliberately conservative rule uses only the pilot's discordance count, not its signed gap. The bound does not make the Gaussian approximation exact and is not a familywise confidence guarantee.
- **known_variance:** oracle comparator using true q - target² at the design alternative. Under a null or half-effect DGP it still plans for the same meaningful effect, not for the realized truth.

A zero-gap or zero-variance plug-in plan is undefined, not a zero-item study. Budgets above 20,000 are unrun, not truncated. Each result records undefined, over-cap and executed counts, budget quantiles among defined plans, and mean executed cost. Workflow rejection counts unrun plans as no rejection. Conditional rejection is restricted to executed plans and can look much better; both denominators are explicit. Unrun comparisons retain p=1 in the complete three-test Holm family.

Within each outer replicate, a new pilot determines each budget, then 32 independent heldout families are drawn without reusing pilot observations. Monte Carlo SEs and approximate 95% intervals use the **outer replicate as the cluster**, including a ratio-of-means delta method for conditional rates. The inner trials are not treated as independent pilots. Zero-event normal intervals can collapse; fixed-N and shared-model rates use Wilson intervals instead. These are pointwise Monte Carlo intervals, not simultaneous guarantees across the grid.

## What the recorded simulation says

At pilot N=128 and the meaningful effect, nominal workflow rejection fractions are:

| DGP | Observed gap | Fixed effect | Variance-only bound | Known variance |
|---|---:|---:|---:|---:|
| q=1%, effect=0.5pp | 14.7% | 59.4% | 95.4% | 76.5% |
| q=5%, effect=2pp | 53.5% | 71.8% | 94.6% | 77.2% |
| q=20%, effect=2pp | 33.3% | 77.8% | 88.4% | 79.2% |
| q=50%, effect=10pp | 67.5% | 77.2% | 83.6% | 77.8% |
| q=80%, effect=60pp | 47.0% | 46.5% | 86.1% | 47.7% |

These percentages are SIM, from `nested.csv`; the complete file includes intervals, null controls, smaller effects and Holm sensitivity. For the q=20%, effect=2pp fixed-effect rule, the nominal estimate is **77.8% [77.4%, 78.2%]**, versus **72.4%** after Holm. For q=1%, the fixed-effect conditional rate is 80.4%, but **785/3,000 plans are unrun**; its workflow rate is only 59.4%. Its bound-rule median defined budget is 11,467 versus 4,867 for the plug-in fixed-effect rule, so the extra detections are not free.

The small-N case is a warning against treating even known variance as a guarantee: the oracle normal budget is 10 items, but the simulated exact-test rejection fraction is only 47.7%. Use the exact reference for discrete small-N designs. Prespecifying the effect removes one large source of optimism, but finite-pilot variance uncertainty and discrete testing still matter. The variance-only bound buys conservatism at a potentially substantial cost; it is not proven optimal or universally calibrated.

## Type-I error and dependent comparisons

Null rows in `reference.csv` and `nested.csv` report type-I behavior, with nominal and Holm family rejection rates in the nested table. The nested comparisons are independent across pairs. A separate panel (`shared_models.csv`) tests all three pairs of three models on the **same items**: a latent easy/hard item has probability 1/2 each and model correctness probabilities .8/.2 conditional on difficulty. A mixed family offsets one model by .1, leaving one true-null pair. Joint eight-cell multinomial sampling preserves the shared-model/item dependence.

At N=1,024 under the global null, the shared-model familywise false-positive fraction is **11.0% [10.6%, 11.4%] nominal** versus **3.77% [3.51%, 4.04%] Holm** (SIM). This is one controlled dependence structure, not coverage of clustered prompts, repeated decodes, distribution shift, model selection, or adaptive stopping. IID items and a prespecified family are still assumptions.

## Measured confirmation sensitivity, kept separate

`confirmation_sensitivity.json` is **MEAS**, computed from the committed repeated-decoding `results/prospective/summary.json` with its source hash. Among 27 executed primary confirmations, there are **23 nominal detections versus 18 after Holm across the 27 tests**. The realized absolute gap distribution is **0.15pp minimum, 11.3pp lower quartile, 16.9pp median, 40.0pp upper quartile, 57.5pp maximum**. Only one comparison is at most 1pp; nine use 16 items. This is mostly evidence about large separations, not prospective validation of hypothetical one-point budgets. The 27-test family is a descriptive sensitivity analysis; it excludes unrun plans and is not a retroactive preregistration.
