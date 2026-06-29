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

## Confidence Interval Heuristic

When the CLT approximation is reasonable and $\sigma$ is estimated by sample standard deviation $s$, an approximate interval for the mean is:

$$
\bar{x}_n
\pm
z_{\alpha/2}
\frac{s}{\sqrt{n}}
$$

This is a finite-sample uncertainty statement about the mean of the sampled population. It is not evidence that the sampled population matches deployment.

## When It Breaks

| Issue | Why CLT intuition can mislead |
| --- | --- |
| correlated examples | effective sample size is smaller than $n$ |
| grouped data | molecules, scaffolds, proteins, prompts, or users are not independent rows |
| heavy tails | rare extreme errors dominate finite samples |
| selected metrics | repeated selection invalidates a naive interval |
| non-average metric | AUROC, top-k enrichment, max score, or filtered generation may need bootstrap or paired methods |

For grouped evaluation, the unit of independence is often the group, not the row:

$$
n_{\mathrm{eff}}
\le
n_{\mathrm{rows}}
$$

If a test set contains many near-duplicates, the apparent standard error can be much too small.

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
- Are rows independent, or should uncertainty be grouped by scaffold, protein family, source, prompt, or user?
- Was the interval computed before or after model selection?

## Related

- [[concepts/math/normal-distribution|Normal distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/statistical-estimator|Statistical estimator]]
- [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/bootstrap-evaluation|Bootstrap evaluation]]
