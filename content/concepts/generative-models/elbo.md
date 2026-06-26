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

## Derivation Sketch

The ELBO follows from inserting $q_\phi(z\mid x)$:

$$
\log p_\theta(x)
=
\log
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right]
$$

and applying Jensen's inequality:

$$
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right]
$$

This can be rearranged into reconstruction and KL terms:

$$
\mathcal{L}_{\mathrm{ELBO}}
=
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))
$$

## KL Gap

The difference between the true log likelihood and ELBO is:

$$
\log p_\theta(x) - \mathcal{L}_{\mathrm{ELBO}}(x)
=
D_{\mathrm{KL}}
\left(q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)\right)
$$

The bound is tight when the approximate posterior matches the true posterior.

## Objective Variants

| Variant | Form | Risk |
| --- | --- | --- |
| negative ELBO | minimize $-\mathcal{L}_{\mathrm{ELBO}}$ | sign conventions can confuse reported losses |
| beta-VAE | reconstruction $-\beta$ KL | disentanglement/regularization trades off likelihood |
| KL annealing | gradually increase KL weight | training schedule is part of the method |
| free bits | lower-bound per-dimension KL contribution | prevents unused latents but changes objective |
| reconstruction-only | drop or weaken KL | good reconstructions may not sample from the prior |

For paper notes, state whether the reported number is ELBO, negative ELBO, reconstruction term, KL term, or a modified surrogate.

## Sampling Boundary

Training uses $q_\phi(z\mid x)$, but unconditional sampling usually uses the prior:

$$
z\sim p(z),
\qquad
\hat{x}\sim p_\theta(x\mid z)
$$

Good reconstructions do not prove good prior samples. For molecules, proteins, sequences, and structures, evaluate generated samples for validity, novelty, diversity, and task utility.

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
- Are reconstruction metrics separated from generation metrics?
- Does the latent variable encode information used by the downstream task?

## Related

- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/latent-variable-model|Latent variable model]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
