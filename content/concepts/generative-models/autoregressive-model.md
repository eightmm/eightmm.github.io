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

## Ordering Contract

Autoregressive modeling needs an ordering:

$$
x = (x_{\pi(1)},\ldots,x_{\pi(T)})
$$

where $\pi$ is a serialization of the object. For text, the ordering is natural. For molecules, graphs, sets, or structures, the ordering is a modeling choice.

| Object | Ordering Choice | Risk |
| --- | --- | --- |
| text | document order | context truncation and prompt leakage |
| protein sequence | residue order | homolog and family leakage |
| molecule SMILES | string traversal | multiple strings for same molecule |
| graph | node/edge serialization | permutation-sensitive likelihood |
| 3D structure | atom/residue order | coordinate validity not guaranteed by token likelihood |

If the object is not naturally sequential, likelihood can partly measure serialization quality rather than object quality.

## Teacher Forcing vs Free Running

Training usually uses teacher forcing:

$$
\log p_\theta(x_{1:T})
=
\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t}^{\mathrm{data}})
$$

Sampling uses generated prefixes:

$$
\hat{x}_t
\sim
p_\theta(\cdot\mid \hat{x}_{<t})
$$

This gap explains why low token NLL can still produce invalid long outputs. Generation notes should report both likelihood-style metrics and sample-level validity or utility.

## Decoding Contract

A generated output is a property of both model and decoder:

$$
\hat{x}
\sim
D(p_\theta,\eta)
$$

where $D$ is the decoding algorithm and $\eta$ contains temperature, top-$k$, top-$p$, beam width, repetition penalties, stopping rules, or validity filters. Comparisons should fix $\eta$ or report sensitivity.

## Why It Matters

- Exact likelihood and stable training make it a default for sequences.
- Backbone of large language models and many token-based generators.
- Sequential sampling can be slow for long outputs.

## Failure Modes

- Exposure bias: training conditions on true prefixes while sampling conditions on generated prefixes.
- Ordering bias: a bad serialization can make local conditionals harder than necessary.
- Degeneration: decoding can repeat, collapse, or produce invalid long-range structure.
- Evaluation mismatch: low NLL does not always imply useful or valid samples.
- Filter dependence: post-processing or rejection sampling may produce the apparent quality.

## Checks

- Does the ordering of elements suit the data?
- Is exposure bias between training and sampling a problem?
- Can sampling be parallelized or distilled for speed?
- Is the decoding rule fixed before comparing models?
- Does the output validity metric match the generated object?
- Are invalid, duplicate, or rejected samples included in the denominator?

## Related

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/llm/decoding|Decoding]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[papers/architectures/pixel-recurrent-neural-networks|PixelRNN / PixelCNN]]
- [[papers/architectures/neural-discrete-representation-learning|VQ-VAE]]
