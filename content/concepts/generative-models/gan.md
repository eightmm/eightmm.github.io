---
title: Generative Adversarial Network
tags:
  - gan
  - generative-model
  - machine-learning
---

# Generative Adversarial Network (GAN)

A GAN trains a generator against a discriminator in a minimax game: the generator produces samples and the discriminator tries to tell them from real data.

The original minimax objective is:

$$
\min_G \max_D
\mathbb{E}_{x\sim p_{\mathrm{data}}}[\log D(x)]
+ \mathbb{E}_{z\sim p(z)}[\log(1-D(G(z)))]
$$

Here $G$ generates samples and $D$ distinguishes real from generated samples.

In practice, the non-saturating generator loss is often used:

$$
\mathcal{L}_G
=
-\mathbb{E}_{z\sim p(z)}[\log D(G(z))]
$$

The discriminator loss is:

$$
\mathcal{L}_D
=
-\mathbb{E}_{x\sim p_{\mathrm{data}}}[\log D(x)]
-\mathbb{E}_{z\sim p(z)}[\log(1-D(G(z)))]
$$

The learned generator defines an implicit distribution because samples are easy but exact likelihood is usually unavailable:

$$
z \sim p(z), \qquad x = G_\theta(z)
$$

## Training Dynamics

GAN training is a two-player optimization problem, not a single static loss. If $D$ becomes too strong, generator gradients can vanish; if $G$ finds a small set of outputs that fool $D$, mode collapse can occur.

Stabilization choices include architecture constraints, gradient penalties, spectral normalization, label smoothing, replay buffers, and careful update ratios.

## Why It Matters

- Can produce sharp, high-fidelity samples without an explicit likelihood.
- Influential in image synthesis and conditional generation.
- Training can be unstable and prone to mode collapse.

## Failure Modes

- Mode collapse: many latent codes map to a narrow output subset.
- Non-convergence: generator and discriminator keep chasing each other.
- Metric mismatch: visually sharp samples may not cover the data distribution.
- Memorization: sample quality can hide nearest-neighbor copying.

## Checks

- Is training balanced between generator and discriminator?
- Are samples diverse, or collapsed onto a few modes?
- How is sample quality measured without a likelihood?
- Are nearest-neighbor and diversity checks reported?
- Does the evaluation distinguish fidelity from coverage?

## Related

- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
