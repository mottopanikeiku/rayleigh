# Method

This page describes how the simulations, the planner and the follow-up experiments work, and the results tables live in [`results/known_truth`](../results/known_truth).

## The simulation

For binary outcomes A and B, let δ = E[A−B] and q = P(A≠B), the fraction of items where exactly one model is right. The four outcomes 00, 01, 10, 11 have probabilities (1−q)/2, (q−δ)/2, (q+δ)/2, (1−q)/2, so Var(A−B) = q−δ².

Five designs set q and the target gap: q=1% with 0.5 points, q=5% with 2, q=20% with 2, q=50% with 10, q=80% with 60. Each runs at a true effect of zero, half the target and the target, with pilot sizes 32, 128 and 512.

Each of 1,000 outer repetitions draws pilots for three independent model pairs. Each pilot picks a budget, then 32 independent evaluation samples test it with an exact two-sided McNemar test at alpha 0.05, which makes 96,000 tests per rule, design, effect and pilot size. A budget that is undefined, or above 20,000 items, counts as a miss. The seed is 20261029 with NumPy's PCG64, and [`configuration.json`](../results/known_truth/configuration.json) has the rest.

The five rules appear in the result files under these names:

| Rule | Budget calculation |
|---|---|
| `observed_gap` | Gaussian planner on the pilot's absolute gap and sample variance |
| `fixed_effect` | Gaussian planner on the target gap and the pilot's variance |
| `discordance_bound` | Target gap, with a one-sided 95% Clopper–Pearson upper bound on q as the variance |
| `known_variance` | Target gap and the true variance q−target² |
| `exact_oracle` | Smallest n whose exact McNemar power reaches 80% at the true q |

The bound uses Var(A−B) ≤ q, so it needs only the discordance count and stays defined when that count is zero. It is Browne's upper-confidence-limit idea [1] applied to binary pairs, and the CLI calls its budget `bound_items_per_model`.

## Exact power

`exact_mcnemar_power(delta, discordance, n, alpha)` averages over the discordance count M ~ Binomial(n, q), and given M=m, the A-only successes are Binomial(m, (q+δ)/(2q)). It sums the probability of the exact two-sided rejection region, including p-values equal to alpha, and [`reference.csv`](../results/known_truth/reference.csv) checks it against 20,000 simulated evaluations at each fixed size.

Even the true-variance Gaussian plan misses 80%, because the planner uses q−δ² for both normal quantiles while McNemar's null variance is q [2]:

| Design | Gaussian, true variance | Exact, true q |
|---|---:|---:|
| q=1%, gap 0.5 points | 3,132 items, 76.5% power | 3,373 items, 80.0% |
| q=5%, gap 2 points | 974 items, 77.3% | 1,035 items, 80.0% |
| q=20%, gap 2 points | 3,917 items, 79.0% | 4,016 items, 80.0% |
| q=50%, gap 10 points | 385 items, 77.6% | 408 items, 80.0% |
| q=80%, gap 60 points | 10 items, 47.5% | 18 items, 82.6% |

At large budgets the exact test's discreteness costs 1–3.5 points of power, while at ten items the variance mismatch is most of the gap. Exact power can drop when n goes up by one, so check nearby budgets with `exact_mcnemar_power` for small studies.

## Several comparisons

Every rule kept its false-positive rate under 5% (worst case 4.69%). Across three independent null comparisons the chance of at least one false positive reached 13.4%, and Holm kept it under 4.8%. The planners target one comparison at a time: at q=20%, a 2-point gap and a 128-item pilot, fixed-gap planning detected 77.8% of effects before Holm and 72.4% after.

## Six small models

Qwen2.5 1.5B, 3B and 7B, SmolLM2 1.7B, Phi-3.5-mini and Mistral-7B-v0.3 answered GSM8K and ARC-Challenge questions, 64 pilot questions per benchmark, once greedily and five times at temperature 0.7. The pilots set the budgets (observed-gap planning) for fresh questions, and 23 of 27 comparisons run on them detected a difference with a paired t-test on five-answer item means, while 17 still did after Holm over all 30 planned comparisons. Three plans needed more than the 512-question collection limit. The median gap was 16.9 points, so this mostly tested large gaps. [`protocol.json`](../results/prospective/protocol.json) fixes the models, prompts and decoding, and [`confirmation_sensitivity.json`](../results/known_truth/confirmation_sensitivity.json) has every gap, adjusted p-value and item count.

Repeated answers split the variance of a difference in model means into an item part and a decoding part, σ²_item + σ²_decode/k, and planning floors negative component estimates at zero.

Two fixes came before the confirmation data, one to grading and one to ARC outputs. GSM8K grading accepts the last number in the answer, because requiring `####` had scored Qwen1.5 at 15.9% instead of 65.9% on the same pilot texts. ARC outputs are restricted to the offered labels with vLLM guided choice, since Phi truncated 18.1% of free-form samples at 1,024 tokens. vLLM's child-seed offsets had also given neighboring questions the same random stream (426 of 3,840 requests), so every item now gets its own seed block, and [`pilot_revision.json`](../results/prospective/pilot_revision.json) holds the before-and-after pilots.

## Arena votes

A Bradley–Terry fit to 54,985 Chatbot Arena votes over 55 models [3] finds 4 of 54 adjacent model pairs with a 95% interval excluding zero, and the median plan for 80% power is 3.3 million votes. For 12 pairs chosen independently of scores, planning from one vote split and resampling another detected a median 96% of the time, with a range of 9–100% ([summary](../results/arena/summary.json), [pairs](../results/arena/pilot_planning.csv)).

## Recompute

```sh
uv sync --locked
uv run python scripts/calibrate.py        # tables and known_truth.svg, no model downloads
uv run python scripts/hero.py             # docs/hero-*.svg from the result files
uv run python scripts/check_results.py    # recompute everything and compare with the saved files
```

`check_results.py` allows 1e-12 relative drift and compares the figures and the example CSV with the saved ones. It also checks simulated rates against exact power, null error rates, Holm families and plan counts, and CI runs it on both x86-64 and arm64.

## References

1. Browne (1995). [On the use of a pilot sample for sample size determination](https://pubmed.ncbi.nlm.nih.gov/8532986/). Statistics in Medicine 14(17).
2. Connor (1987). [Sample size for testing differences in proportions for the paired-sample design](https://pubmed.ncbi.nlm.nih.gov/3567305/). Biometrics 43(1).
3. Chiang et al. (2024). [Chatbot Arena](https://arxiv.org/abs/2403.04132). Vote source and license: [`data/arena_manifest.json`](../data/arena_manifest.json).

Related work on paired power for LLM evaluation includes [Miller 2024](https://arxiv.org/html/2411.00640), [Card et al. 2020](https://aclanthology.org/2020.emnlp-main.745/), [Madaan et al. 2024](https://arxiv.org/html/2406.10229), [tinyBenchmarks](https://proceedings.mlr.press/v235/maia-polo24a.html), [Kotawala 2026](https://arxiv.org/html/2605.30315) and [Basile et al. 2026](https://aclanthology.org/2026.lrec-1.353/). For planning from noisy pilots, see [Albers and Lakens 2018](https://doi.org/10.1016/j.jesp.2017.09.004) and [Kraemer et al. 2006](https://pubmed.ncbi.nlm.nih.gov/16651505/).
