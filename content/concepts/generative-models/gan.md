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

This means evaluation must rely on samples, discriminators, nearest-neighbor checks, or downstream utility rather than exact density:

$$
x_1,\ldots,x_n \sim p_\theta
\quad\text{but}\quad
\log p_\theta(x)\ \text{is usually unavailable}
$$

## Training Dynamics

GAN training is a two-player optimization problem, not a single static loss. If $D$ becomes too strong, generator gradients can vanish; if $G$ finds a small set of outputs that fool $D$, mode collapse can occur.

Stabilization choices include architecture constraints, gradient penalties, spectral normalization, label smoothing, replay buffers, and careful update ratios.

## Fidelity-Coverage Boundary

GAN evaluation must separate fidelity from coverage.

| Axis | Question | Failure if ignored |
| --- | --- | --- |
| fidelity | Do samples look or score like real examples? | sharp but memorized or invalid samples |
| coverage | Does the generator cover the data modes? | mode collapse |
| novelty | Are samples distinct from training examples? | nearest-neighbor copying |
| conditional control | Does the sample satisfy condition $c$? | condition leakage or ignored condition |
| efficiency | How many candidate samples are needed per useful sample? | hidden rejection or cherry-picking |

For conditional GANs:

$$
z\sim p(z),
\qquad
x = G_\theta(z,c),
\qquad
D_\psi(x,c)\in[0,1]
$$

The condition $c$ must be part of the evaluation. A sample that fools an unconditional discriminator may still fail the requested class, scaffold, pose, or property.

## Stabilization Patterns

| Pattern | Purpose | Caveat |
| --- | --- | --- |
| non-saturating loss | avoid weak generator gradients | still sensitive to discriminator balance |
| gradient penalty | regularize discriminator smoothness | changes training cost and objective |
| spectral normalization | bound discriminator Lipschitz behavior | can limit capacity if too restrictive |
| minibatch or diversity features | expose collapse to discriminator | may not cover rare modes |
| two-time-scale updates | balance $G$ and $D$ learning speed | update ratio becomes part of the method |

## Why It Matters

- Can produce sharp, high-fidelity samples without an explicit likelihood.
- Influential in image synthesis and conditional generation.
- Training can be unstable and prone to mode collapse.

## Failure Modes

- Mode collapse: many latent codes map to a narrow output subset.
- Non-convergence: generator and discriminator keep chasing each other.
- Metric mismatch: visually sharp samples may not cover the data distribution.
- Memorization: sample quality can hide nearest-neighbor copying.
- Hidden filtering: only a subset of generated samples is shown or scored.
- Conditional failure: samples look realistic but ignore the requested condition.

## Checks

- Is training balanced between generator and discriminator?
- Are samples diverse, or collapsed onto a few modes?
- How is sample quality measured without a likelihood?
- Are nearest-neighbor and diversity checks reported?
- Does the evaluation distinguish fidelity from coverage?
- Are attempted samples, rejected samples, and shown samples counted separately?
- Is the discriminator used only for training, or also as a reported evaluator?
- For conditional generation, is condition satisfaction measured independently?

## Related

- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
