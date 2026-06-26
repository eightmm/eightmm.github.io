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

## Generative Story

A latent-variable paper should make the generative story explicit:

$$
z \sim p(z),
\qquad
x \sim p_\theta(x\mid z)
$$

For conditional generation:

$$
z \sim p(z\mid c)
\quad\text{or}\quad
z\sim p(z),
\qquad
x\sim p_\theta(x\mid z,c)
$$

These two forms make different assumptions: the condition can shape the latent prior, the decoder, or both.

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

In variational models, this leads to the evidence lower bound:

$$
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))
$$

The reconstruction term, KL term, prior, and posterior family all define what the latent is allowed to encode.

## Latent Design Axes

| Axis | Choices | Risk |
| --- | --- | --- |
| type | continuous, discrete, categorical, graph, sequence, hierarchy | wrong type makes latent hard to use |
| prior | standard normal, learned prior, conditional prior, diffusion prior | sampling from prior may not match encoded latents |
| posterior | exact, amortized, variational, encoder-based, MCMC | posterior approximation can dominate performance |
| decoder | likelihood model, autoregressive decoder, diffusion decoder, graph decoder | strong decoder can ignore $z$ |
| use | compression, controllability, uncertainty, diversity, representation | latent claim may not match evaluation |

## Failure Modes

| Failure | Symptom | Check |
| --- | --- | --- |
| posterior collapse | decoder ignores $z$ | mutual information, KL term, latent traversals, ablation |
| prior mismatch | encoded latents decode well but prior samples fail | compare reconstruction and prior-sample validity |
| entangled latent | changing one latent changes many factors | controlled intervention or conditional evaluation |
| leakage latent | latent encodes source, split, label, or template artifact | source/split-stratified analysis |
| invalid latent region | interpolation or random samples decode invalid objects | validity over prior and interpolation paths |

## Checks

- What does the latent variable represent?
- Is $z$ discrete, continuous, structured, or hierarchical?
- Is the latent space used by the decoder or ignored?
- Can sampling from $p(z)$ produce valid outputs?
- Does the latent variable improve controllability or only add complexity?
- Is evaluation based on reconstructions, prior samples, interpolations, or optimized latents?
- Does the paper distinguish latent representation quality from generated sample quality?
- Is the latent chosen, searched, or filtered at inference time?

## Related

- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/elbo|Evidence lower bound]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/guidance|Guidance]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/math/probability-distribution|Probability distribution]]
