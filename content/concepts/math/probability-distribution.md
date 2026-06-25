---
title: Probability Distribution
tags:
  - math
  - probability
---

# Probability Distribution

A probability distribution assigns probabilities to possible outcomes. In machine learning, data, labels, latent variables, model outputs, noise, and deployment environments are often described as distributions.

For a discrete variable:

$$
\sum_x p(x) = 1,\qquad p(x)\ge 0
$$

For a continuous variable:

$$
\int p(x)\,dx = 1,\qquad p(x)\ge 0
$$

## Joint, Marginal, and Conditional

Joint distribution:

$$
p(x,y)
$$

Marginal distribution:

$$
p(x)=\sum_y p(x,y)
$$

Conditional distribution:

$$
p(y\mid x)=\frac{p(x,y)}{p(x)}
$$

The product rule factorizes a joint distribution:

$$
p(x,y)=p(y\mid x)p(x)=p(x\mid y)p(y)
$$

For a sequence $x_{1:T}$, the chain rule is:

$$
p(x_{1:T})
=
\prod_{t=1}^{T}
p(x_t\mid x_{<t})
$$

This is the probability factorization behind autoregressive modeling.

## Model vs Data Distribution

Machine learning often compares the data distribution and a model distribution:

$$
p_{\mathrm{data}}(x,y)
\quad\text{and}\quad
p_\theta(y\mid x)
\text{ or }
p_\theta(x)
$$

Evaluation asks whether performance under a held-out or deployment distribution supports the intended claim.

## Checks

- What random variable does the distribution describe?
- Is the distribution over inputs, labels, latents, outputs, or noise?
- Is the model estimating $p(x)$, $p(y\mid x)$, $p(x,y)$, or a score?
- Is the variable discrete, continuous, structured, or mixed?
- Does the training distribution match the deployment distribution?
- Is the factorization assumption explicit?

## Related

- [[concepts/math/random-variable|Random variable]]
- [[concepts/math/normal-distribution|Normal distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/math/bayes-rule|Bayes rule]]
- [[concepts/machine-learning/density-estimation|Density estimation]]
- [[concepts/generative-models/index|Generative models]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
