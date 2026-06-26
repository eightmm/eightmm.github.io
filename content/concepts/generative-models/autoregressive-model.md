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

Training usually maximizes the likelihood of the observed sequence, or equivalently minimizes [[concepts/machine-learning/negative-log-likelihood|negative log-likelihood]]:

$$
\mathcal{L}_{\mathrm{NLL}}
=
- \sum_{t=1}^{T}
\log p_\theta(x_t \mid x_{<t})
$$

For a Transformer decoder, the conditional distribution is typically:

$$
h_t = f_\theta(x_{<t}), \qquad
p_\theta(x_t \mid x_{<t})
=
\operatorname{softmax}(W_o h_t)_{{x_t}}
$$

where $h_t$ is the hidden state and $W_o$ maps it to vocabulary logits.

## Sampling

Sampling applies a decoding rule to the conditional distribution:

$$
x_t \sim q(\cdot \mid x_{<t})
$$

where $q$ may be the original model distribution, temperature-scaled probabilities, top-$k$, nucleus sampling, greedy decoding, or beam search. The decoding rule changes diversity and failure modes, so it belongs to the model contract.

## Why It Matters

- Exact likelihood and stable training make it a default for sequences.
- Backbone of large language models and many token-based generators.
- Sequential sampling can be slow for long outputs.

## Failure Modes

- Exposure bias: training conditions on true prefixes while sampling conditions on generated prefixes.
- Ordering bias: a bad serialization can make local conditionals harder than necessary.
- Degeneration: decoding can repeat, collapse, or produce invalid long-range structure.
- Evaluation mismatch: low NLL does not always imply useful or valid samples.

## Checks

- Does the ordering of elements suit the data?
- Is exposure bias between training and sampling a problem?
- Can sampling be parallelized or distilled for speed?
- Is the decoding rule fixed before comparing models?
- Does the output validity metric match the generated object?

## Related

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/llm/decoding|Decoding]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
