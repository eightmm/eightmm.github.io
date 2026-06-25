---
title: Autoregressive Model
tags:
  - autoregressive-model
  - generative-model
  - machine-learning
---

# Autoregressive Model

An autoregressive model factorizes the joint distribution as a product of conditionals, generating one element at a time conditioned on previous ones.

The core factorization is:

$$
p_\theta(x)
= \prod_{t=1}^{T}
p_\theta(x_t \mid x_{<t})
$$

Training usually maximizes the likelihood of the observed sequence, or equivalently minimizes [[concepts/machine-learning/negative-log-likelihood|negative log-likelihood]].

## Why It Matters

- Exact likelihood and stable training make it a default for sequences.
- Backbone of large language models and many token-based generators.
- Sequential sampling can be slow for long outputs.

## Checks

- Does the ordering of elements suit the data?
- Is exposure bias between training and sampling a problem?
- Can sampling be parallelized or distilled for speed?

## Related

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
