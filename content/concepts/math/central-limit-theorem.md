---
title: Central Limit Theorem
tags:
  - math
  - probability
  - statistics
---

# Central Limit Theorem

The central limit theorem explains why averages of many independent random variables often look approximately normal, even when the original variables are not normal.

Let $x_1,\ldots,x_n$ be independent samples with mean $\mu$ and variance $\sigma^2$. The sample mean is:

$$
\bar{x}_n
=
\frac{1}{n}\sum_{i=1}^{n}x_i
$$

The normalized mean converges in distribution:

$$
\frac{\sqrt{n}(\bar{x}_n-\mu)}{\sigma}
\Rightarrow
\mathcal{N}(0,1)
$$

where $\Rightarrow$ denotes convergence in distribution.

## Standard Error

The variance of the sample mean decreases with $n$:

$$
\operatorname{Var}(\bar{x}_n)
=
\frac{\sigma^2}{n}
$$

The standard error is:

$$
\operatorname{SE}(\bar{x}_n)
=
\frac{\sigma}{\sqrt{n}}
$$

This is why larger evaluation sets can reduce metric noise, though they do not fix [[concepts/evaluation/leakage|leakage]] or [[concepts/data/dataset-shift|dataset shift]].

## Why It Matters

- It justifies many approximate confidence intervals for averages.
- It explains why repeated noisy measurements can become stable when averaged.
- It connects metric uncertainty to sample size.
- It does not guarantee that biased samples represent the deployment distribution.

## Checks

- Are examples independent enough for the approximation to be meaningful?
- Is the sample size large relative to skew, heavy tails, or correlation?
- Is the reported uncertainty about finite-sample noise, not dataset bias?
- Is the metric an average, or a nonlinear statistic requiring another method such as bootstrap?

## Related

- [[concepts/math/normal-distribution|Normal distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/statistical-estimator|Statistical estimator]]
- [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/bootstrap-evaluation|Bootstrap evaluation]]
