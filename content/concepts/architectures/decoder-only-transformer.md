---
title: Decoder-Only Transformer
tags:
  - architectures
  - transformer
  - autoregressive-model
---

# Decoder-Only Transformer

A decoder-only Transformer uses causal self-attention and predicts the next token autoregressively. It is the standard architecture pattern for many large language models and sequence generators.

Canonical paper notes in this wiki include [[papers/architectures/gpt-2|GPT-2]], [[papers/architectures/gpt-3|GPT-3]], and [[papers/architectures/llama|LLaMA]].

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

At layer $\ell$, causal self-attention computes:

$$
Q = H^{(\ell)}W_Q,\quad
K = H^{(\ell)}W_K,\quad
V = H^{(\ell)}W_V
$$

$$
\operatorname{Attn}(H^{(\ell)})
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)V
$$

where $M$ prevents a token from attending to future tokens. During inference, previously computed keys and values can be cached:

$$
K_{\le t} = [K_{<t}; K_t],\quad
V_{\le t} = [V_{<t}; V_t]
$$

This makes incremental decoding avoid recomputing the whole prefix, but memory grows with context length, layers, heads, and batch size.

## Training and Inference

- Training usually uses teacher forcing: every position predicts the next token in parallel under the causal mask.
- Inference samples or decodes one step at a time, often with KV cache.
- Long-context behavior depends on [[concepts/architectures/positional-encoding|positional encoding]], attention implementation, and memory budget.
- Valid output may require constrained decoding for molecules, tool calls, code, or structured actions.

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
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[infra/gpu/index#memory|GPU memory]]
- [[agents/tools/tool-use|Tool use]]
