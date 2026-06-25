---
title: Latent Variable Model
tags:
  - generative-models
  - latent-variable
  - machine-learning
---

# Latent Variable Model

A latent variable model introduces an unobserved variable $z$ that explains or controls observed data $x$.

The marginal likelihood is:

$$
p_\theta(x)
=
\int p_\theta(x \mid z)p(z)\,dz
$$

$p(z)$ is the prior over latent variables and $p_\theta(x\mid z)$ is the decoder or likelihood model.

## Why It Matters

Latent variables can represent factors of variation, compressed structure, uncertainty, or hidden causes. They appear in [[concepts/generative-models/vae|VAE]] models, hierarchical generators, diffusion latents, and conditional generation systems.

## Inference

The posterior over latents is:

$$
p_\theta(z \mid x)
=
\frac{p_\theta(x \mid z)p(z)}{p_\theta(x)}
$$

When exact inference is hard, models use approximate inference:

$$
q_\phi(z \mid x) \approx p_\theta(z \mid x)
$$

## Checks

- What does the latent variable represent?
- Is $z$ discrete, continuous, structured, or hierarchical?
- Is the latent space used by the decoder or ignored?
- Can sampling from $p(z)$ produce valid outputs?
- Does the latent variable improve controllability or only add complexity?

## Related

- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
