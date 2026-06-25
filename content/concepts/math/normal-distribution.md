---
title: Normal Distribution
tags:
  - math
  - probability
  - statistics
---

# Normal Distribution

The normal distribution is a continuous probability distribution defined by a mean $\mu$ and variance $\sigma^2$. It is used as a noise model, an approximation from the [[concepts/math/central-limit-theorem|central limit theorem]], and a building block for regression and uncertainty estimates.

For $x \in \mathbb{R}$:

$$
x \sim \mathcal{N}(\mu,\sigma^2)
$$

with density:

$$
p(x)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right)
$$

where $\mu=\mathbb{E}[x]$ and $\sigma^2=\mathbb{E}[(x-\mu)^2]$.

## Standardization

A normal variable can be standardized:

$$
z
=
\frac{x-\mu}{\sigma}
$$

If $x\sim\mathcal{N}(\mu,\sigma^2)$, then $z\sim\mathcal{N}(0,1)$.

## Multivariate Normal

For a vector $x\in\mathbb{R}^d$:

$$
x \sim \mathcal{N}(\mu,\Sigma)
$$

where $\mu$ is a mean vector and $\Sigma$ is a covariance matrix. This connects to [[concepts/math/covariance-correlation|covariance and correlation]].

## Why It Matters

- Gaussian negative log-likelihood gives [[concepts/machine-learning/mean-squared-error|mean squared error]] when variance is fixed.
- Many confidence intervals use a normal approximation.
- Latent-variable models often use Gaussian priors or approximate posteriors.
- Regression models may predict both mean and variance for uncertainty.

## Checks

- Is the variable continuous and approximately symmetric?
- Is variance fixed, estimated, or predicted?
- Are tails important enough that a Gaussian assumption may be unsafe?
- Is independence assumed across dimensions, or is a full covariance matrix needed?

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/math/central-limit-theorem|Central limit theorem]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
