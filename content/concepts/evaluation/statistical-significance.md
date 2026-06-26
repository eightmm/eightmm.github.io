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
