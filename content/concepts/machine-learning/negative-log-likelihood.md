---
title: Negative Log-Likelihood
tags:
  - machine-learning
  - loss
  - probability
---

# Negative Log-Likelihood

Negative log-likelihood trains a probabilistic model by penalizing low probability assigned to observed data.

For a conditional model $p_\theta(y\mid x)$ and dataset $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{n}$:

$$
\mathcal{L}_{\mathrm{NLL}}(\theta)
=
-
\frac{1}{n}
\sum_{i=1}^{n}
\log p_\theta(y_i\mid x_i)
$$

This is equivalent to maximum likelihood estimation:

$$
\hat{\theta}_{\mathrm{MLE}}
=
\arg\max_\theta
\sum_{i=1}^{n}
\log p_\theta(y_i\mid x_i)
$$

## Common Cases

Categorical likelihood gives [[concepts/machine-learning/cross-entropy-loss|cross-entropy loss]]:

$$
\mathcal{L}_{\mathrm{NLL}}
=
-
\log p_\theta(y=c\mid x)
$$

Gaussian likelihood with fixed variance gives [[concepts/machine-learning/mean-squared-error|mean squared error]] up to constants and scale:

$$
\mathcal{L}_{\mathrm{NLL}}
=
\frac{1}{2\sigma^2}
(y-\mu_\theta(x))^2
+
C
$$

Autoregressive sequence models factorize likelihood over positions:

$$
p_\theta(x_{1:T})
=
\prod_{t=1}^{T}
p_\theta(x_t\mid x_{<t})
$$

so the sequence NLL is:

$$
\mathcal{L}_{\mathrm{NLL}}
=
-
\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t})
$$

## Checks

- What probability distribution is assumed: categorical, Bernoulli, Gaussian, mixture, or autoregressive?
- Are log probabilities averaged per example, per token, per atom, or per sequence?
- Are padding, masks, and invalid positions excluded correctly?
- Is NLL used for training, evaluation, or both?
- Does likelihood align with downstream utility, validity, ranking, or calibration?

## Related

- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/cross-entropy-loss|Cross-entropy loss]]
- [[concepts/machine-learning/mean-squared-error|Mean squared error]]
- [[concepts/machine-learning/density-estimation|Density estimation]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/llm/language-model|Language model]]
