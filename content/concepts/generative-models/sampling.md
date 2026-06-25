---
title: Sampling
tags:
  - generative-models
  - inference
  - sampling
---

# Sampling

Sampling is the procedure that turns a learned generative model into concrete outputs.

For an unconditional model:

$$
x \sim p_\theta(x)
$$

For a conditional model:

$$
x \sim p_\theta(x \mid c)
$$

The sampling algorithm determines quality, diversity, latency, and controllability.

## Sampling Patterns

- Autoregressive sampling: draw one token or element at a time.
- Diffusion sampling: iteratively denoise from a noise distribution.
- Flow sampling: transform a base sample through an invertible or learned transport map.
- Latent sampling: sample $z$ first, then decode $x$.
- Rejection or filtering: generate candidates and keep those passing validity or score checks.

## Diversity and Quality

Sampling often trades diversity for fidelity. If the sampler collapses to high-probability modes, outputs may look good but cover only a small part of the data distribution.

## Checks

- What distribution is sampled first: token, noise, latent, graph, or coordinates?
- How many sampling steps or decoding calls are required?
- Are temperature, guidance strength, beam size, or rejection filters documented?
- Are invalid samples counted or silently removed?
- Are diversity, novelty, and task utility evaluated separately?

## Related

- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/latent-variable-model|Latent variable model]]
- [[concepts/llm/decoding|Decoding]]
