---
title: Long Short-Term Memory
aliases:
  - papers/lstm
  - papers/long-short-term-memory
tags:
  - papers
  - architectures
  - recurrent
---

# Long Short-Term Memory

> The paper introduced gated recurrent memory cells designed to preserve error signals over long time spans.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Long Short-Term Memory |
| Authors | Sepp Hochreiter, Jurgen Schmidhuber |
| Year | 1997 |
| Venue | Neural Computation |
| DOI | [10.1162/neco.1997.9.8.1735](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory) |
| Status | verified |

## Question

Vanilla recurrent networks struggled when useful information had to cross many time steps. The question was whether a recurrent architecture could keep a stable internal memory while still being trained by gradient descent.

## Main Claim

Long short-term memory uses gated memory cells to reduce vanishing-gradient failure in sequence learning.

Modern notation:

$$
\begin{aligned}
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) \\
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) \\
\tilde{c}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

The architectural point is not just recurrence, but a protected state path $c_t$ controlled by gates.

## Method

| Component | Role |
| --- | --- |
| memory cell | stores information across time |
| input gate | controls writing into memory |
| output gate | controls exposure of memory to hidden state |
| recurrent state | carries temporal context |
| constant error path | mitigates rapid gradient decay |

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| LSTM can bridge long time lags | synthetic long-delay tasks | original tasks are small by modern standards |
| Gated memory helps gradient flow | training dynamics and benchmark comparisons | later LSTM variants added forget gates and other changes |
| Recurrent memory is architecture-level bias | sequence-learning experiments | Transformers later changed the default sequence modeling recipe |

## Limitations

- The widely used forget-gate LSTM differs from the original 1997 version.
- Sequential recurrence limits parallelism over time.
- Long-range memory is still capacity-limited and can be hard to optimize.
- For many modern language and vision tasks, attention-based models replaced LSTM as the default.

## Why It Matters

LSTM is the canonical gated recurrent architecture and remains the cleanest reference point for memory, recurrence, and vanishing-gradient discussions.

## Connections

- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/lstm|LSTM]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[papers/architectures/layer-normalization|Layer Normalization]]
- [[papers/architectures/index|Architecture papers]]
