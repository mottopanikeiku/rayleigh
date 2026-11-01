# Related work and experiment details

Paired power analysis is well established. Rayleigh tests how much power survives when a small pilot supplies the planning inputs, then compares those results with evaluations on new questions.

## Paired evaluation and small benchmarks

Miller [1] derives sample-size calculations for paired model differences and discusses clustered items. Card et al. [4] simulate paired-accuracy power from the accuracy gap and model agreement, using McNemar tests. Both make the covariance between models' scores part of the calculation.

Estimating a benchmark score and detecting a small difference need different item budgets. tinyBenchmarks [3] estimates full-benchmark scores with curated items and item-response models, reporting roughly two-percentage-point average error with 100 items per scenario. Madaan et al. [2] examine 13 benchmarks and more than 280 models or checkpoints; their small-benchmark experiments can preserve means while increasing variability and reversing rankings. Blackwell et al. [8] study prediction intervals across repeated evaluation runs.

Kotawala [5] provides paired minimum-detectable-effect and item-count diagnostics, exact and approximate McNemar tests, multiplicity adjustment and subject clustering. Its validation resamples three model pairs using parameters estimated from the full response matrix. Basile et al. [7] provide paired simulation-based planning through sk-power and analyze more than 2,000 comparisons across five datasets. Neuhof and Benjamini [6] build Holm-adjusted rank intervals and show why subject heterogeneity and selection after ranking matter.

## Planning from a pilot

Albers and Lakens [9] show how noisy pilot effects and selective follow-up can leave main studies underpowered. They recommend choosing the smallest effect of interest independently of the pilot. Kraemer et al. [10] discuss the same problem in study proposals. Browne's upper-confidence-limit approach to pilot variance is the basis for the discordance-bound rule described in [the simulation methods](CALIBRATION.md).

For aligned scores, let $D_i=X_{Ai}-X_{Bi}$. With independent items,

$$
SE_{paired}^2=s_D^2/n,\qquad SE_{unpaired}^2=(s_A^2+s_B^2)/n.
$$

Positive covariance reduces the paired variance. Choose a meaningful gap $\delta_0$, significance level $\alpha$ and target power $1-\beta$ before estimating variance. The usual normal approximation is

$$
n\approx (z_{1-\alpha/2}+z_{1-\beta})^2\sigma_D^2/\delta_0^2.
$$

For binary scores, discordance $q=P(D\ne0)$ gives $Var(D)=q-\delta^2$. Exact McNemar power averages the conditional binomial test over the possible discordance counts. The Python API includes both the Gaussian planner and the exact calculation.

When questions share a subject or prompt, the independent-item calculation can understate uncertainty. Resample aligned subject groups and decide whether the target is a fixed set of subjects or a wider subject population. For several model comparisons, define the full family before applying Holm; selecting adjacent models after ranking does not remove the selection problem [5,6].

## Historical benchmark analysis

The response matrices cover 395 models from the 2024 Open LLM Leaderboard across ARC-Challenge, GSM8K, HellaSwag, MMLU and WinoGrande. I split items into pilot and evaluation sets, then compare predicted power with rejection rates on the unused items. The [calculator](https://mottopanikeiku.github.io/rayleigh/) shows this analysis alongside its planner. [Settings](../results/configuration.json) and [summary results](../results/calibration.csv) give the sampling details.

The analysis also separates resampling with replacement from recovery of a finite benchmark's mean. For $N$ fixed observed differences, a simple random sample without replacement has variance

$$
Var(\bar D_n\mid D_{1:N})=(1-n/N)S_D^2/n,
$$

where $S_D^2$ uses denominator $N-1$. That finite-pool correction describes the existing benchmark. New subjects and new model responses introduce additional uncertainty.

## Repeated answers

I generated answers from six instruction-tuned models on GSM8K and ARC-Challenge. The [protocol](../results/prospective/protocol.json) specifies model revisions, prompts, item IDs and decoding settings. Each model answered 64 pilot questions per benchmark greedily and with five stochastic draws.

For the difference between models' mean scores on one question, I estimate

$$
\operatorname{Var}(\bar Y_A-\bar Y_B)=\sigma^2_{\mathrm{item}}+\sigma^2_{\mathrm{decode}}/k.
$$

Decoding variance comes from repeated answers to each question. Subtracting it from the variance of item means estimates the item component. The results include negative component estimates; planning uses a zero floor. Averaging over n independent questions divides this variance by n.

The [sample-size plans](../results/prospective/primary/plan.json) were set after the pilots and before collecting the confirmation answers. Each feasible pair received 16–264 new questions and a paired t-test on five-answer means. Comparisons sharing a model reused its answers. The collection produced 9,222 confirmation answers. Three plans exceeded the 512-question limit: GSM8K Qwen3–Phi needed 870; ARC Qwen1.5–Mistral needed 1,503 and Qwen3–Phi needed 881.

### Grading and seed fixes

GSM8K uses final-number extraction inspired by lm-evaluation-harness [12]. The first grader required `####`; accepting the final number changed Qwen1.5's score from 15.9% to 65.9% on the same pilot texts. For ARC, Phi truncated 18.1% of unconstrained samples at 1,024 tokens. I switched all six models to vLLM guided choice [13], restricting output to `Answer: <one of the provided labels>`.

vLLM's child-seed offset [14] also caused neighboring questions to reuse 426 of 3,840 requested random streams. I assigned a separate seed block to each item and reran both pilots with 3,840 distinct stochastic seeds. The [pilot comparison](../results/prospective/pilot_revision.json) contains the earlier results and the effects of these changes. Strict-format rescoring of confirmation answers detects 22 of 27 differences, compared with 23 under the primary grader.

The [collector](../modal_app.py) saves answers, seeds, model revisions and generation settings. Estimated cloud cost, including setup failures and diagnostic reruns, was at most $3.02 from the [allocated resources](../results/prospective/costs.json).

## Preference votes

I also fit a Bradley–Terry model to historical Chatbot Arena votes [11]. After filtering by model exposure, 54,985 votes cover 55 models. Both inverse-Hessian and bootstrap 95% intervals exclude zero for only 4 of 54 adjacent model pairs. The median fitted requirement for 80% power is 3.30 million total votes across the comparison network.

For 12 pairs chosen independently of scores, I planned from one vote split and ran 200 resamples of another. The median detection rate was 96.25%, with a range of 9–100%. The llama-2-7b-chat / llama2-70b-steerlm-chat pair required 142,611 network votes in the plan but detected a difference in only 18 of 200 resamples. [Summary](../results/arena/summary.json) · [Pair results](../results/arena/pilot_planning.csv) · [Methods](../results/arena/methods.txt).

## Limitations

- Historical matrices and votes describe the model populations and questions in those releases. Public benchmark contamination is possible.
- Repeated-answer results depend on the grading and generation method. ARC guided choice differs from option-likelihood scoring, and Qwen2.5-3B has a noncommercial research license.
- Fewer questions with repeated decoding can still require more generated answers and GPU time.
- The Arena model assumes independent votes and a single transitive preference strength. User and prompt clustering are unavailable in the aggregates; adjacent-pair intervals have no simultaneous or rank-selection adjustment.

## References

1. Evan Miller. 2024. *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations*. [arXiv](https://arxiv.org/html/2411.00640).
2. Lovish Madaan et al. 2024. *Quantifying Variance in Evaluation Benchmarks*. NeurIPS Regulatable ML workshop. [Paper](https://arxiv.org/html/2406.10229) · [Venue](https://neurips.cc/virtual/2024/106953).
3. Felipe Maia Polo et al. 2024. *tinyBenchmarks: evaluating LLMs with fewer examples*. ICML, PMLR 235:34303–34326. [Proceedings](https://proceedings.mlr.press/v235/maia-polo24a.html).
4. Dallas Card et al. 2020. *With Little Power Comes Great Responsibility*. EMNLP, pp. 9263–9274. [Paper](https://aclanthology.org/2020.emnlp-main.745/).
5. Anany Kotawala. 2026. *Resolution Diagnostics for Paired LLM Evaluation*. [arXiv](https://arxiv.org/html/2605.30315) · [Implementation](https://github.com/akotawala10/llm-power).
6. Bitya Neuhof and Yuval Benjamini. 2026. *Quantifying Ranking Uncertainty in LLM Benchmarks*. [arXiv](https://arxiv.org/html/2607.16259).
7. Angelo Basile, Areg Mikael Sarvazyan and José Ángel González. 2026. *How Many Samples Do We Need? A Toolkit for Power-Aware Evaluation Design*. LREC, pp. 4507–4513. [Paper](https://aclanthology.org/2026.lrec-1.353/).
8. Robert E. Blackwell, Jon Barry and Anthony G. Cohn. 2025 revision. *Towards Reproducible LLM Evaluation: Quantifying Uncertainty in LLM Benchmark Scores*. [arXiv](https://arxiv.org/html/2410.03492v2).
9. Casper Albers and Daniël Lakens. 2018. *When power analyses based on pilot data are biased: Inaccurate effect size estimators and follow-up bias*. Journal of Experimental Social Psychology 74:187–195. [Paper](https://doi.org/10.1016/j.jesp.2017.09.004).
10. Helena Chmura Kraemer et al. 2006. *Caution regarding the use of pilot studies to guide power calculations for study proposals*. Archives of General Psychiatry 63(5):484–489. [PubMed](https://pubmed.ncbi.nlm.nih.gov/16651505/).
11. Wei-Lin Chiang et al. 2024. *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*. [arXiv](https://arxiv.org/abs/2403.04132) · [Vote source and license](../data/arena_manifest.json).
12. EleutherAI. *lm-evaluation-harness*. [GSM8K grading configuration](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml).
13. vLLM 0.10.2. [Guided-choice documentation](https://docs.vllm.ai/en/v0.10.2/serving/openai_compatible_server.html#extra-parameters).
14. vLLM 0.10.2. [Parallel sampling implementation](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/engine/parallel_sampling.py).
