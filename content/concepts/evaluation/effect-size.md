---
title: Effect Size
tags:
  - evaluation
  - statistics
  - methodology
---

# Effect Size

Effect size measures how large a difference is, not only whether it is statistically detectable. In model evaluation, it answers whether a reported improvement is large enough to matter for the task, cost, or scientific claim.

For a metric $M$ where higher is better, the raw effect is:

$$
\Delta M = M(f_A) - M(f_B)
$$

$f_A$ is the proposed model, $f_B$ is the baseline, and $M$ is the metric under the same evaluation protocol.

## Standardized Effect

When metrics vary across tasks or folds, a standardized effect can be useful:

$$
d = \frac{\bar{x}_A - \bar{x}_B}{s_{\mathrm{pooled}}}
$$

where $\bar{x}_A$ and $\bar{x}_B$ are mean scores and $s_{\mathrm{pooled}}$ is a pooled standard deviation. This is only meaningful when the score units and independence assumptions are clear.

## Practical Importance

An effect is meaningful when:

$$
|\Delta M| > \delta_{\mathrm{min}}
$$

$\delta_{\mathrm{min}}$ is the smallest improvement that would change a decision: deploy, reproduce, reject, collect more data, or investigate a mechanism.

## Decision Boundary

Effect size should be interpreted against a decision, not only against zero.

| Decision | Effect Question |
| --- | --- |
| deploy | Is the gain large enough to justify latency, cost, risk, and maintenance? |
| reproduce | Is the effect large enough that a rerun is worth the compute? |
| scientific claim | Is the effect consistent with the proposed mechanism? |
| benchmark claim | Does the gain survive baseline, split, and seed checks? |
| data collection | Would more labels or better labels change the conclusion? |

For paired evaluation on the same examples, inspect per-example differences:

$$
\Delta_i = m_i(f_A)-m_i(f_B)
$$

The average effect $\bar{\Delta}$ can hide that gains are concentrated in one subset while another subset gets worse.

## Checks

- Is the effect larger than the confidence interval or seed variance?
- Is the effect measured on the final test set, not the validation set used for selection?
- Does the effect change decisions, cost, robustness, calibration, or scientific interpretation?
- Is the metric scale interpretable for the task?
- Is the effect consistent across important strata, not only the aggregate score?
- Is the effect reported together with uncertainty and the number of comparisons attempted?

## Common Mistakes

- Reporting only a $p$-value without the metric difference.
- Treating a tiny gain on a huge benchmark as practically important.
- Ignoring negative effects on rare classes, OOD subsets, validity, latency, or calibration.
- Comparing effect sizes across benchmarks with different metrics and split policies.
- Using validation-selected gains as if they were final-test effects.

## Related

- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/seed-variance|Seed variance]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/research-methodology/hypothesis|Hypothesis]]
- [[papers/analysis/claim-extraction|Claim extraction]]
