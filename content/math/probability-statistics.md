---
title: Probability and Statistics
tags:
  - math
  - probability
  - statistics
---

# Probability and Statistics

Probability describes uncertainty and data-generating processes. Statistics describes how finite samples estimate unknown quantities.

$$
\mathbb{E}_{(x,y)\sim p_{\mathrm{data}}}
[\mathcal{L}(f(x), y)]
$$

The distribution under the expectation matters as much as the loss itself.

## Probability

- [[concepts/math/random-variable|Random variable]]
- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/normal-distribution|Normal distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/bayes-rule|Bayes rule]]

## Distribution Map

| Quantity | Meaning | Common AI Use |
| --- | --- | --- |
| $p(x)$ | marginal distribution of inputs | data modeling, density estimation |
| $p(y\mid x)$ | conditional label distribution | classification, regression uncertainty |
| $p_\theta(y\mid x)$ | model-predicted conditional distribution | probabilistic prediction and calibration |
| $p_{\mathrm{train}}(x,y)$ | training distribution | empirical risk minimization |
| $p_{\mathrm{test}}(x,y)$ | evaluation distribution | generalization claim |
| $q(z\mid x)$ | approximate posterior or encoder distribution | VAE, latent-variable inference |

Bayes rule connects posterior, likelihood, and prior:

$$
p(y\mid x)
=
\frac{p(x\mid y)p(y)}{p(x)}
$$

This is useful when reading probabilistic classifiers, latent-variable models, and uncertainty notes.

## Statistics

- [[concepts/math/statistical-estimator|Statistical estimator]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/math/central-limit-theorem|Central limit theorem]]
- [[concepts/math/hypothesis-testing|Hypothesis testing]]
- [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]]
- [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]]

## Estimation

A finite dataset gives an estimate of a population quantity:

$$
\mu = \mathbb{E}_{X\sim p}[h(X)],
\qquad
\hat{\mu}
=
\frac{1}{n}\sum_{i=1}^{n} h(x_i)
$$

The error is controlled by sampling noise, bias, dependence between samples, and whether the sample distribution matches the target distribution.

For an estimator $\hat{\theta}$:

$$
\operatorname{MSE}(\hat{\theta})
=
\operatorname{Var}(\hat{\theta})
+
\operatorname{Bias}(\hat{\theta})^2
$$

## AI Connections

- Probabilistic prediction needs calibrated probabilities, not only scores.
- Dataset shift changes the distribution under the expected risk.
- Uncertainty estimation depends on what randomness is being modeled.
- Hypothesis testing and confidence intervals help interpret benchmark differences.

## Checks

- Is the probability conditional or marginal?
- Is the estimate biased, high-variance, or data-leaking?
- Is the test distribution the same as the deployment distribution?
- Are repeated evaluations creating multiple-comparison risk?

## Related

- [[math/index|Math]]
- [[ai/evaluation|Evaluation]]
- [[concepts/evaluation/calibration|Calibration]]
