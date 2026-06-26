---
title: Density Estimation
tags:
  - machine-learning
  - generative-models
---

# Density Estimation

Density estimation learns or approximates the probability distribution that generated data. It is the statistical root of many generative models.

The target is a distribution $p_{\mathrm{data}}(x)$, while the model provides $p_\theta(x)$ or an approximation to it. Learning usually tries to make $p_\theta$ assign high probability to training examples and low probability to implausible regions.

The maximum-likelihood objective is:

$$
\hat{\theta}
= \arg\max_\theta
\sum_{i=1}^{n}
\log p_\theta(x_i)
$$

Equivalently, it minimizes [[concepts/machine-learning/negative-log-likelihood|negative log-likelihood]]:

$$
\mathcal{L}_{\mathrm{NLL}}
= -\frac{1}{n}\sum_{i=1}^{n}\log p_\theta(x_i)
$$

In expectation form, maximum likelihood minimizes cross entropy:

$$
H(p_{\mathrm{data}},p_\theta)
=
-\mathbb{E}_{x\sim p_{\mathrm{data}}}
\left[\log p_\theta(x)\right]
$$

Because

$$
H(p_{\mathrm{data}},p_\theta)
=
H(p_{\mathrm{data}})
+
D_{\mathrm{KL}}(p_{\mathrm{data}}\Vert p_\theta),
$$

the model is pushed toward the data distribution when the model class and optimization are adequate.

## Common Forms

- Explicit density models.
- Autoregressive factorization.
- Normalizing flows.
- Variational latent-variable models.
- Score or denoising objectives that avoid direct density evaluation.

Autoregressive models factorize density as:

$$
p_\theta(x)
=
\prod_{t=1}^{T}
p_\theta(x_t\mid x_{<t})
$$

Normalizing flows use an invertible transform $x=f_\theta(z)$:

$$
\log p_\theta(x)
=
\log p(z)
-
\log
\left|
\det
\frac{\partial f_\theta(z)}{\partial z}
\right|
$$

Score-based models learn the score instead of the density value:

$$
s_\theta(x,t)
\approx
\nabla_x \log p_t(x)
$$

## Density vs Generation

Likelihood and sample quality are related but not identical. A model can assign high likelihood to common low-level statistics while generating poor samples, or generate useful samples while making exact likelihood unavailable.

For conditional generation, the target is:

$$
p_\theta(x\mid c)
$$

where $c$ can be a class label, text prompt, protein context, molecular scaffold, or other conditioning signal.

## Checks

- Is likelihood tractable, estimated, bounded, or unavailable?
- Does high likelihood correspond to useful sample quality?
- Is the modeled distribution conditional or unconditional?
- Are out-of-distribution samples assigned misleadingly high density?
- Is evaluation using held-out likelihood, sample quality, downstream utility, or calibration?
- Are discrete, continuous, and structured outputs handled with compatible density assumptions?

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/machine-learning/cross-entropy-loss|Cross-entropy loss]]
- [[concepts/generative-models/index|Generative models]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/score-based-model|Score-based model]]
