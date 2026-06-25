---
title: LSTM
tags:
  - architectures
  - rnn
  - sequence-modeling
---

# LSTM

Long short-term memory networks are gated [[concepts/architectures/rnn|RNNs]] designed to carry information over longer sequences. They maintain a hidden state $h_t$ and a cell state $c_t$.

The standard update is:

$$
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
$$

$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
$$

$$
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
$$

$$
\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

The forget gate $f_t$ controls retained memory, the input gate $i_t$ controls new content, and the output gate $o_t$ controls exposed state.

## Key Ideas

- LSTMs reduce the vanishing-gradient problem by giving memory an additive path through $c_t$.
- They are sequential at inference and training unless special approximations are used.
- They remain useful baselines for streaming, low-latency, and smaller sequence tasks.
- In modern papers, they often appear as historical baselines against [[concepts/architectures/transformer|Transformers]] or [[concepts/architectures/state-space-model|state-space models]].

## Practical Checks

- Check whether the model is unidirectional or bidirectional.
- Track whether the output uses $h_T$, pooled states, or token-level hidden states.
- Inspect truncation length for backpropagation through time.
- Compare against [[concepts/architectures/gru|GRU]] when parameter count and speed matter.

## Related

- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/gru|GRU]]
- [[concepts/architectures/state-space-model|State-space models]]
- [[concepts/architectures/transformer|Transformer]]
