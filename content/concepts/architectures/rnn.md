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

## Key Ideas

- RNNs update a hidden state one step at a time, so order is built into the computation.
- The hidden state is a bottleneck that must carry useful past information forward.
- Gated variants add learned mechanisms for retaining and forgetting information.
- Bidirectional RNNs use past and future context for representation tasks, but not for strict causal generation.
- RNNs are useful comparisons for [[concepts/architectures/state-space-model|state-space models]] because both scan sequences through state updates.

## Practical Checks

- Check whether the task needs causal, bidirectional, or streaming inference.
- Watch sequence length, truncation, masking, and padding behavior.
- Inspect how the final representation is chosen: last state, pooled states, attention over states, or token-level outputs.
- Compare with [[concepts/architectures/transformer|Transformers]] when long-range interactions or parallel training are central.

## Related

- [[concepts/architectures/state-space-model|State-space models]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/learning/index|Learning methods]]
