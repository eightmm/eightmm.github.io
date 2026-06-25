---
title: Density Estimation
tags:
  - machine-learning
  - generative-models
---

# Density Estimation

Density estimation learns or approximates the probability distribution that generated data. It is the statistical root of many generative models.

The maximum-likelihood objective is:

$$
\hat{\theta}
= \arg\max_\theta
\sum_{i=1}^{n}
\log p_\theta(x_i)
$$

Equivalently, it minimizes negative log-likelihood:

$$
\mathcal{L}_{\mathrm{NLL}}
= -\frac{1}{n}\sum_{i=1}^{n}\log p_\theta(x_i)
$$

## Common Forms

- Explicit density models.
- Autoregressive factorization.
- Normalizing flows.
- Variational latent-variable models.
- Score or denoising objectives that avoid direct density evaluation.

## Checks

- Is likelihood tractable, estimated, bounded, or unavailable?
- Does high likelihood correspond to useful sample quality?
- Is the modeled distribution conditional or unconditional?
- Are out-of-distribution samples assigned misleadingly high density?

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/generative-models/index|Generative models]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/vae|VAE]]
