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

## Objective Contract

| Term | Formula | Meaning | Risk |
| --- | --- | --- | --- |
| Reconstruction | $\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]$ | decoder explains observed data | may reward token-level fidelity but not utility |
| Prior matching | $D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))$ | encoder posterior stays close to prior | too strong can erase information |
| Sampling | $z\sim p(z),\ x\sim p_\theta(x\mid z)$ | generate from prior | prior samples can be less valid than reconstructions |
| Inference | $z\sim q_\phi(z\mid x)$ | encode observed data | representation may not be useful for downstream tasks |

When reading a VAE paper, separate reconstruction evidence from generation evidence:

$$
x \rightarrow z \rightarrow \hat{x}
\qquad \text{is not the same as} \qquad
z\sim p(z) \rightarrow x
$$

Good reconstructions do not prove that prior samples are valid, diverse, novel, or useful.

## Latent Diagnostics

| Diagnostic | What It Checks |
| --- | --- |
| KL per dimension | whether only a few latent coordinates carry information |
| Mutual-information proxy | whether $z$ depends on $x$ |
| Prior-sample validity | whether random $z\sim p(z)$ decodes to valid objects |
| Reconstruction validity | whether encoded examples decode correctly |
| Interpolation path | whether latent interpolation stays inside valid regions |
| Downstream probe | whether $z$ captures the property claimed by the paper |

Posterior collapse can be described as:

$$
q_\phi(z\mid x) \approx p(z)
$$

which means the latent variable carries little input-specific information.

## Computational Biology Use

| Object | Decoder Output | Extra Check |
| --- | --- | --- |
| molecule string | SMILES or token sequence | chemical validity, uniqueness, scaffold novelty |
| molecular graph | atom and bond graph | valence, charge, stereochemistry, disconnected fragments |
| conformer | coordinates | chirality, bond geometry, conformer energy |
| protein sequence | amino acid sequence | length, motif, family leakage, function evidence |
| structure or complex | coordinates, distances, contacts | equivariance, residue mapping, pose quality |

For molecule and protein generation, a VAE result should not stop at ELBO. It should also report validity, diversity, novelty, and the downstream task boundary.

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
- Are reconstruction, representation, and generation claims evaluated separately?
- If used for molecules or proteins, are invalid samples counted in the denominator?

## Related

- [[concepts/architectures/autoencoder|Autoencoder]]
- [[concepts/generative-models/latent-variable-model|Latent variable model]]
- [[concepts/generative-models/elbo|Evidence lower bound]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/gan|GAN]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/molecular-generation|Molecular generation]]
