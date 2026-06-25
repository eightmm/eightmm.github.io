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

## Why It Matters

- Provides a structured, continuous latent space for sampling and interpolation.
- A building block in many molecular and protein generators.
- Trades off reconstruction quality against latent regularization.

## Checks

- Is the latent space collapsing or being ignored by the decoder?
- Does the reconstruction–KL balance match the goal?
- Are latent samples decoding to valid, diverse outputs?

## Related

- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/gan|GAN]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
