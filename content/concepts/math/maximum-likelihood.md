---
title: Maximum Likelihood
tags:
  - math
  - likelihood
  - machine-learning
---

# Maximum Likelihood

Maximum likelihood chooses model parameters that make the observed data likely under the model.

For independent examples:

$$
\theta^\star
= \arg\max_\theta
\prod_{i=1}^{n} p_\theta(x_i)
$$

It is usually optimized as log-likelihood:

$$
\theta^\star
= \arg\max_\theta
\sum_{i=1}^{n}\log p_\theta(x_i)
$$

or negative log-likelihood:

$$
\mathcal{L}_{\mathrm{NLL}}
= -\sum_{i=1}^{n}\log p_\theta(x_i)
$$

As an empirical expectation:

$$
\mathcal{L}_{\mathrm{NLL}}
=
-n\,
\mathbb{E}_{x\sim \hat{p}_{\mathcal{D}}}
\left[
\log p_\theta(x)
\right]
$$

where $\hat{p}_{\mathcal{D}}$ is the empirical data distribution.

For supervised learning, the conditional version is:

$$
\mathcal{L}_{\mathrm{NLL}}
= -\sum_{i=1}^{n}\log p_\theta(y_i\mid x_i)
$$

For sequence modeling with the chain rule:

$$
-\log p_\theta(x_{1:T})
=
-\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t})
$$

## Why It Matters

- Cross-entropy classification is conditional maximum likelihood.
- Autoregressive language modeling minimizes next-token negative log-likelihood.
- Density estimation and many generative models are likelihood-based.
- Likelihood is not always aligned with sample quality or downstream utility.
- Maximizing likelihood is equivalent to minimizing cross-entropy from the data distribution to the model distribution.

## Checks

- What probability is being maximized: $p(x)$, $p(y\mid x)$, or $p(x,y)$?
- Is the likelihood exact, approximated, bounded, or implicit?
- Does high likelihood correspond to the task metric?
- Are examples assumed independent when they are grouped or duplicated?
- Is the objective token-level, example-level, trajectory-level, or structure-level?

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/machine-learning/density-estimation|Density estimation]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
