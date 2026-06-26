---
title: Variational Autoencoder
tags:
  - vae
  - generative-model
  - representation-learning
---

# Variational Autoencoder (VAE)

A VAE learns a latent-variable generative model by jointly training an encoder and decoder to maximize a variational lower bound on the data likelihood.

The evidence lower bound is:

$$
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
- D_{\mathrm{KL}}(q_\phi(z\mid x)\,\Vert\,p(z))
$$

The first term rewards reconstruction; the KL term regularizes the latent distribution.

The generative process and inference model are:

$$
z \sim p(z), \qquad
x \sim p_\theta(x\mid z), \qquad
z \sim q_\phi(z\mid x)
$$

The reparameterization trick makes gradients flow through stochastic latent samples:

$$
z = \mu_\phi(x) + \sigma_\phi(x)\odot \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I)
$$

For a $\beta$-VAE style objective:

$$
\mathcal{L}
=
\mathbb{E}_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
- \beta D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))
$$

where $\beta$ controls the reconstruction-latent regularization tradeoff.

## Why It Matters

- Provides a structured, continuous latent space for sampling and interpolation.
- A building block in many molecular and protein generators.
- Trades off reconstruction quality against latent regularization.

## Failure Modes

- Posterior collapse: the decoder ignores $z$ and the latent carries little information.
- Over-regularization: samples are smooth but reconstructions are poor.
- Invalid decoding: latent samples map outside the valid output space.
- Latent interpolation can look meaningful without matching downstream utility.

## Checks

- Is the latent space collapsing or being ignored by the decoder?
- Does the reconstruction–KL balance match the goal?
- Are latent samples decoding to valid, diverse outputs?
- Is validity evaluated on prior samples, not only reconstructions?
- Is the decoder likelihood appropriate for the data type?

## Related

- [[concepts/architectures/autoencoder|Autoencoder]]
- [[concepts/generative-models/latent-variable-model|Latent variable model]]
- [[concepts/generative-models/elbo|Evidence lower bound]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/gan|GAN]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
