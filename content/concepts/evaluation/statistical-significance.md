---
title: Statistical Significance
tags:
  - evaluation
  - statistics
  - methodology
---

# Statistical Significance

Statistical significance asks whether an observed difference is larger than what could plausibly arise from evaluation noise under a null hypothesis.

The mathematical testing procedure is summarized in [[concepts/math/hypothesis-testing|hypothesis testing]]. This note focuses on how that idea should be interpreted in model evaluation.

For two models $A$ and $B$, the important quantity is often the metric difference:

$$
\Delta M
=
M_A - M_B
$$

A hypothesis test compares:

$$
H_0:\Delta M = 0
\qquad
\text{vs.}
\qquad
H_1:\Delta M \ne 0
$$

A $p$-value is the probability, under $H_0$, of seeing a result at least as extreme as the observed statistic:

$$
p
=
P(|T|\ge |T_{\mathrm{obs}}| \mid H_0)
$$

## Key Ideas

- Significance does not measure effect size or practical importance.
- A tiny improvement can be statistically significant on a huge benchmark but irrelevant in practice.
- A useful improvement can be statistically inconclusive if the benchmark is too small or noisy.
- Paired tests are usually preferable when two models are evaluated on the same examples.
- Multiple comparisons increase false positives if many variants are tested and only the best result is reported.

## Practical Checks

- What is the unit of independence: example, scaffold, protein family, target, run, or benchmark?
- Is the comparison paired on the same test cases?
- Is the effect size meaningful even if statistically significant?
- Were many models, seeds, thresholds, or prompts tried before reporting the winner?

## Evaluation Interpretation

| Situation | Better Reading |
| --- | --- |
| significant but tiny effect | true under the tested protocol, but maybe not worth deploying or discussing |
| nonsignificant but wide interval | benchmark may be underpowered; absence of evidence is not evidence of equality |
| significant after many variants | check multiple comparisons and selection bias |
| significant aggregate, subgroup failures | narrow the claim and report failure modes |
| significant on validation set | not final evidence if validation selected the model |

## Test Choice

| Setting | Typical Choice | Watch |
| --- | --- | --- |
| same examples, scalar metric | paired bootstrap or paired test | resampling unit must match dependence |
| classification counts | McNemar-style paired error test or bootstrap | class imbalance and threshold selection |
| ranking/retrieval | query-level bootstrap | candidate-level resampling is invalid |
| many random seeds | repeated-run analysis | best seed is not expected performance |
| many datasets/tasks | per-dataset paired effect | average can hide negative transfer |

## Claim Contract

A statistical significance statement should include:

| Field | Required Detail |
| --- | --- |
| null hypothesis | what equality or no-effect claim is being tested |
| statistic | metric difference, error difference, rank difference, or effect size |
| unit | example, query, scaffold, target, family, seed, prompt, or dataset |
| selection boundary | whether model, checkpoint, prompt, threshold, and preprocessing were fixed before final test |
| multiplicity | number of tested variants, metrics, slices, and baselines |
| practical threshold | smallest effect that would matter for the claim |

## Related

- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/effect-size|Effect size]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/multiple-comparisons|Multiple comparisons]]
- [[concepts/math/hypothesis-testing|Hypothesis testing]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
