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

## Checks

- Is the effect larger than the confidence interval or seed variance?
- Is the effect measured on the final test set, not the validation set used for selection?
- Does the effect change decisions, cost, robustness, calibration, or scientific interpretation?
- Is the metric scale interpretable for the task?
- Is the effect consistent across important strata, not only the aggregate score?

## Common Mistakes

- Reporting only a $p$-value without the metric difference.
- Treating a tiny gain on a huge benchmark as practically important.
- Ignoring negative effects on rare classes, OOD subsets, validity, latency, or calibration.
- Comparing effect sizes across benchmarks with different metrics and split policies.

## Related

- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/research-methodology/hypothesis|Hypothesis]]
- [[papers/claim-extraction|Claim extraction]]
