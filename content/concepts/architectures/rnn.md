---
title: Recurrent Neural Networks
tags:
  - architectures
  - rnn
  - sequence-modeling
---

# Recurrent Neural Networks

Recurrent neural networks process sequences through a carried hidden state. They are an older sequence-modeling family and a useful reference point for [[concepts/architectures/state-space-model|state-space models]].

The basic recurrence is:

$$
h_t = \phi(W_x x_t + W_h h_{t-1} + b)
$$

The hidden state $h_t$ is the compressed memory of the prefix up to time $t$.

For sequence-to-sequence outputs, the recurrence is often paired with an output projection:

$$
y_t = W_y h_t + b_y
$$

For sequence-level prediction, the model may use $h_T$, a pooled sequence of hidden states, or an attention/readout layer over $\{h_t\}_{t=1}^{T}$.

## Key Ideas

- RNNs update a hidden state one step at a time, so order is built into the computation.
- The hidden state is a bottleneck that must carry useful past information forward.
- Gated variants such as [[concepts/architectures/lstm|LSTM]] and [[concepts/architectures/gru|GRU]] add learned mechanisms for retaining and forgetting information.
- Bidirectional RNNs use past and future context for representation tasks, but not for strict causal generation.
- RNNs are useful comparisons for [[concepts/architectures/state-space-model|state-space models]] because both scan sequences through state updates.

## Sequence Contract

| Field | Question |
| --- | --- |
| Direction | causal, bidirectional, or encoder-only? |
| State size | how much memory can $h_t$ carry? |
| Output | token-level, sequence-level, pooled, or autoregressive? |
| Masking | how are padding and variable lengths handled? |
| Truncation | full BPTT or truncated backpropagation through time? |
| Streaming | can inference process one token at a time? |

For long sequences, the hidden state is both the memory and the bottleneck:

$$
h_t = F_\theta(x_t, h_{t-1})
$$

so information from $x_1$ must survive repeated updates to affect $y_T$.

## Complexity

For sequence length $L$ and hidden width $d$, recurrent computation is usually:

$$
O(Ld^2)
$$

for dense recurrent matrices, with sequential dependence over $t$. This makes RNNs efficient for streaming but harder to parallelize across time than Transformers.

## Transformer-Era Recurrent Models

Modern recurrent language models try to keep the streaming advantage while recovering Transformer-era training scale:

$$
\text{parallel training}
\quad+\quad
\text{recurrent inference state}.
$$

[[papers/architectures/rwkv|RWKV]] is the canonical paper note here. It can be formulated in a Transformer-like parallel mode for training and an RNN-like recurrent mode for inference:

$$
s_t = F_\theta(s_{t-1}, x_t),
\qquad
y_t=G_\theta(s_t,x_t).
$$

The practical tradeoff is:

| Benefit | Risk |
| --- | --- |
| constant-size recurrent state at inference | compressed state may bottleneck retrieval-heavy tasks |
| no growing KV cache | less direct pairwise token access than attention |
| streaming-friendly generation | training and implementation details are architecture-specific |

## Paper Reading Boundary

| Claim | What To Check |
| --- | --- |
| long-context modeling | truncation length and gradient stability |
| streaming inference | latency, state carry-over, reset policy |
| compact baseline | parameter count and hidden size |
| protein sequence modeling | homolog split, sequence identity, length handling |
| time-series or trajectory | sampling rate, missing data, and episode boundary |

## Practical Checks

- Check whether the task needs causal, bidirectional, or streaming inference.
- Watch sequence length, truncation, masking, and padding behavior.
- Inspect how the final representation is chosen: last state, pooled states, attention over states, or token-level outputs.
- Compare with [[concepts/architectures/transformer|Transformers]] when long-range interactions or parallel training are central.
- Is hidden state reset per example, carried across chunks, or initialized from context?
- Are sequence-level metrics separated from token-level metrics?

## Related

- [[concepts/architectures/state-space-model|State-space models]]
- [[concepts/architectures/lstm|LSTM]]
- [[concepts/architectures/gru|GRU]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[papers/architectures/rwkv|RWKV]]
- [[concepts/learning/index|Learning methods]]
