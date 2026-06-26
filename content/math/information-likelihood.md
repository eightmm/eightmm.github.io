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

## Core Quantities

| Quantity | Formula | Use |
| --- | --- | --- |
| Negative log-likelihood | $-\log p_\theta(x)$ | fit probability model to observed data |
| Cross-entropy | $H(p,q)=-\mathbb{E}_{x\sim p}\log q(x)$ | supervised classification, distribution matching |
| Entropy | $H(p)=-\mathbb{E}_{x\sim p}\log p(x)$ | uncertainty of a distribution |
| KL divergence | $D_{\mathrm{KL}}(p\|q)=\mathbb{E}_{x\sim p}\log \frac{p(x)}{q(x)}$ | compare distributions asymmetrically |
| ELBO | $\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))$ | train latent-variable models |

For a labeled dataset, cross-entropy is the negative log-likelihood of the observed class:

$$
\mathcal{L}_{\mathrm{CE}}
=
-\sum_{k=1}^{K} y_k \log p_\theta(y=k\mid x)
$$

where $y_k$ is a one-hot or soft target.

## Generative Models

- [[concepts/generative-models/latent-variable-model|Latent variable model]]
- [[concepts/generative-models/elbo|Evidence lower bound]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]

## Reading Generative Objectives

| Objective Type | Learns | Watch |
| --- | --- | --- |
| Likelihood | normalized probability or tractable factorization | high likelihood may not imply useful samples |
| Variational bound | latent encoder and decoder | bound tightness and posterior assumptions matter |
| Denoising / score | gradient or denoising target over noisy data | sampling path and noise schedule define behavior |
| Flow matching | velocity field along a probability path | path choice and ODE integration affect samples |
| Adversarial | generator through discriminator feedback | mode collapse and unstable training |

## Checks

- What distribution is being modeled?
- Is the loss a likelihood, variational bound, denoising target, score target, or velocity target?
- Are probabilities used for ranking, calibration, sampling, or decision-making?
- Does lower loss imply better downstream utility for the task?

## Related

- [[math/index|Math]]
- [[ai/generative-models|Generative Models]]
- [[ai/learning-methods|Learning Methods]]
