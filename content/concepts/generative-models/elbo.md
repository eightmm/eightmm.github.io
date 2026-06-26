---
title: Evidence Lower Bound
tags:
  - generative-models
  - vae
  - latent-variable
---

# Evidence Lower Bound

The evidence lower bound (ELBO) is a tractable lower bound on the marginal log likelihood of a latent-variable model.

For latent variable $z$ and observation $x$:

$$
\log p_\theta(x)
=
\log \int p_\theta(x,z)\,dz
$$

Directly computing this integral is often hard, so a variational distribution $q_\phi(z\mid x)$ is introduced:

$$
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}
\left(q_\phi(z\mid x)\,\Vert\,p(z)\right)
$$

The first term rewards reconstruction or likelihood under the decoder. The KL term keeps the approximate posterior close to the prior.

## KL Gap

The difference between the true log likelihood and ELBO is:

$$
\log p_\theta(x) - \mathcal{L}_{\mathrm{ELBO}}(x)
=
D_{\mathrm{KL}}
\left(q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)\right)
$$

The bound is tight when the approximate posterior matches the true posterior.

## Why It Matters

- Defines the training objective for VAEs and many latent-variable models.
- Separates reconstruction quality from latent regularization.
- Makes posterior approximation an explicit modeling choice.
- Provides a lens for posterior collapse, weak latents, and overly powerful decoders.

## Checks

- Is the decoder so strong that it ignores $z$?
- Is the KL term weighted, annealed, free-bits controlled, or otherwise modified?
- Does sampling from the prior produce valid outputs, not only reconstructions?
- Is the reported objective an ELBO, negative ELBO, reconstruction-only loss, or a modified objective?

## Related

- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/latent-variable-model|Latent variable model]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
