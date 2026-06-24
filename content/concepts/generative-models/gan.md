---
title: Generative Adversarial Network
tags:
  - gan
  - generative-model
  - machine-learning
---

# Generative Adversarial Network (GAN)

A GAN trains a generator against a discriminator in a minimax game: the generator produces samples and the discriminator tries to tell them from real data.

## Why It Matters

- Can produce sharp, high-fidelity samples without an explicit likelihood.
- Influential in image synthesis and conditional generation.
- Training can be unstable and prone to mode collapse.

## Checks

- Is training balanced between generator and discriminator?
- Are samples diverse, or collapsed onto a few modes?
- How is sample quality measured without a likelihood?

## Related

- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
