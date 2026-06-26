---
title: Information and Likelihood
tags:
  - math
  - likelihood
  - information-theory
---

# Information and Likelihood

Likelihood and information-theoretic quantities connect probability models, loss functions, generative models, and representation learning.

$$
\hat{\theta}
=
\arg\max_\theta
\sum_{i=1}^{n}\log p_\theta(x_i)
$$

Maximum likelihood trains a model to assign high probability to observed data under the modeling assumptions.

## Core Notes

- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/machine-learning/cross-entropy-loss|Cross-entropy loss]]
- [[concepts/machine-learning/probabilistic-prediction|Probabilistic prediction]]
- [[concepts/evaluation/proper-scoring-rule|Proper scoring rule]]

## Generative Models

- [[concepts/generative-models/latent-variable-model|Latent variable model]]
- [[concepts/generative-models/elbo|Evidence lower bound]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]

## Checks

- What distribution is being modeled?
- Is the loss a likelihood, variational bound, denoising target, score target, or velocity target?
- Are probabilities used for ranking, calibration, sampling, or decision-making?
- Does lower loss imply better downstream utility for the task?

## Related

- [[math/index|Math]]
- [[ai/generative-models|Generative Models]]
- [[ai/learning-methods|Learning Methods]]
