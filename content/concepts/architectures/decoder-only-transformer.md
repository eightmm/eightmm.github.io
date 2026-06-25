---
title: Decoder-Only Transformer
tags:
  - architectures
  - transformer
  - autoregressive-model
---

# Decoder-Only Transformer

A decoder-only Transformer uses causal self-attention and predicts the next token autoregressively. It is the standard architecture pattern for many large language models and sequence generators.

The causal mask is:

$$
M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

The sequence likelihood factorizes as:

$$
p_\theta(x_{1:T})
= \prod_{t=1}^{T}
p_\theta(x_t \mid x_{<t})
$$

## Uses

- Text generation.
- Autoregressive protein or molecular sequence generation.
- Tool-call and agent action generation.
- Next-token pretraining.

## Checks

- Does the task require generation or only representation?
- Is context truncated or cached during inference?
- Are generated tokens constrained to valid syntax, molecules, or actions?
- Is evaluation separated into likelihood, sample quality, and task utility?

## Related

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[agents/tool-use|Tool use]]
