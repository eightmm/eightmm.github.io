---
title: Confidence Interval
tags:
  - evaluation
  - statistics
  - uncertainty
---

# Confidence Interval

A confidence interval summarizes uncertainty in an estimated metric. A point estimate such as accuracy, RMSE, enrichment, or success rate is incomplete without asking how much it would vary under another sample or another run.

For a mean score over $n$ independent examples:

$$
\hat{\mu}
=
\frac{1}{n}\sum_{i=1}^{n} x_i
$$

An approximate normal confidence interval is:

$$
\hat{\mu}
\pm
z_{\alpha/2}
\frac{\hat{\sigma}}{\sqrt{n}}
$$

where $\hat{\sigma}$ is the sample standard deviation and $z_{\alpha/2}$ is a normal quantile.

When the metric is not a simple mean, bootstrap resampling is often more practical:

$$
\hat{M}^{(b)}
=
M(\mathcal{D}_{\mathrm{test}}^{(b)})
$$

The interval is estimated from the empirical quantiles of bootstrap metric values $\{\hat{M}^{(b)}\}_{b=1}^{B}$.

## Key Ideas

- A confidence interval describes estimator uncertainty, not guaranteed model behavior on every future case.
- Wide intervals mean the test set, seed count, or benchmark is too small to support a strong claim.
- Overlapping intervals do not automatically prove no difference; paired tests or bootstrap differences are often better.
- For model comparisons, estimate the uncertainty of $\Delta M$, not only the uncertainty of each model separately.

## Paired Difference

For two models evaluated on the same examples, estimate uncertainty on the paired difference:

$$
d_i = m_A(z_i) - m_B(z_i)
$$

$$
\bar{d}
=
\frac{1}{n}\sum_{i=1}^{n} d_i
$$

Then compute an interval for $\bar{d}$ or bootstrap the paired test set. This is usually more informative than comparing two independent confidence intervals.

## What Population Varies?

| Uncertainty Source | What Changes | Typical Report |
| --- | --- | --- |
| example sampling | test examples | bootstrap CI over examples |
| random seed | initialization, order, stochastic training | mean and interval over seeds |
| split choice | train/test partition | grouped CV or repeated split interval |
| annotator or assay noise | measured label | label-aware uncertainty or sensitivity analysis |
| prompt or decoding randomness | sampled generation or agent run | run-level interval and failure rate |

Do not mix these sources without saying what the interval means.

## Small Sample Warning

For rare events or small benchmark slices, normal intervals can be misleading. Prefer bootstrap, exact/binomial intervals, or report the raw numerator and denominator:

$$
\hat{p}
=
\frac{s}{n}
$$

where $s$ is the number of successes and $n$ is the number of evaluated cases.

## Practical Checks

- Is uncertainty across examples, random seeds, data splits, or annotators?
- If uncertainty is across random runs, should it be treated as [[concepts/evaluation/seed-variance|seed variance]] rather than only sampling error?
- Is the metric paired by the same test examples across models?
- Are reported improvements larger than the interval width?
- Is the interval computed on the held-out set, not on validation data used for selection?
- Is the interval for the metric itself or for the improvement over a baseline?
- Are subgroup or target-level intervals needed before making broad claims?

## Related

- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/bootstrap-evaluation|Bootstrap evaluation]]
- [[concepts/evaluation/seed-variance|Seed variance]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/effect-size|Effect size]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
