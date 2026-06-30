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

The density is:

$$
p(x)
=
\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\left(
-\frac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu)
\right)
$$

The quadratic term is the squared Mahalanobis distance:

$$
d_\Sigma^2(x,\mu)
=
(x-\mu)^\top\Sigma^{-1}(x-\mu)
$$

If $\Sigma=\sigma^2 I$, all dimensions have the same independent noise scale. If $\Sigma$ is diagonal, dimensions are independent but can have different variances. A full covariance matrix models correlations, but it is more expensive and harder to estimate.

## Gaussian Negative Log-Likelihood

For a target $y$ and predicted mean $\mu_\theta(x)$ with fixed variance $\sigma^2$:

$$
-\log p(y\mid x)
=
\frac{1}{2\sigma^2}
\|y-\mu_\theta(x)\|_2^2
+ \text{constant}
$$

So minimizing Gaussian negative log-likelihood with fixed variance is equivalent to minimizing mean squared error up to scale and constants.

If the model also predicts variance $\sigma_\theta^2(x)$:

$$
-\log p(y\mid x)
=
\frac{1}{2}
\left[
\log \sigma_\theta^2(x)
+
\frac{(y-\mu_\theta(x))^2}{\sigma_\theta^2(x)}
\right]
+ \text{constant}
$$

This can represent heteroscedastic uncertainty, but the variance prediction must be checked with calibration, not only training loss.

## Sampling and Reparameterization

A Gaussian sample can be written as:

$$
x = \mu + \sigma \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,1)
$$

For a multivariate Gaussian with covariance $\Sigma = LL^\top$:

$$
x = \mu + L\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I)
$$

This form appears in latent-variable models and differentiable sampling because randomness is isolated in $\epsilon$.

## When the Assumption Breaks

| Assumption | Failure Mode |
| --- | --- |
| symmetric residuals | skewed errors make mean and interval claims misleading |
| light tails | outliers dominate squared-error training |
| independent dimensions | diagonal covariance underestimates joint uncertainty |
| fixed variance | high-noise and low-noise regions receive the same confidence |
| calibrated variance | predicted uncertainty is interpreted as probability without validation |

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
- If variance is predicted, is uncertainty calibrated on held-out data?
- Is the normal distribution used as a data model, residual approximation, latent prior, or estimator approximation?

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/math/central-limit-theorem|Central limit theorem]]
- [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/machine-learning/mean-squared-error|Mean squared error]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/calibration|Calibration]]
