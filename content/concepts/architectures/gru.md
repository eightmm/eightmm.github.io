---
title: GRU
tags:
  - architectures
  - rnn
  - sequence-modeling
---

# GRU

Gated recurrent units are gated [[concepts/architectures/rnn|RNNs]] with fewer gates than [[concepts/architectures/lstm|LSTMs]]. They use update and reset gates to control hidden-state memory.

The update is:

$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$

$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

$$
\tilde{h}_t = \tanh(W_h x_t + U_h(r_t \odot h_{t-1}) + b_h)
$$

$$
h_t = (1-z_t)\odot h_{t-1} + z_t\odot \tilde{h}_t
$$

The update gate $z_t$ controls how much new candidate state replaces old memory.

## Key Ideas

- GRUs are simpler than LSTMs because they merge cell and hidden state.
- They are often faster and smaller than LSTMs while retaining useful gating.
- They still process sequences recurrently, so long-range interaction is mediated through a hidden-state bottleneck.
- They are useful as baselines for streaming or compact sequence models.

## Gate Interpretation

| Gate | Role |
| --- | --- |
| update gate $z_t$ | interpolates old state and new candidate state |
| reset gate $r_t$ | controls how much previous state contributes to candidate state |
| candidate $\tilde{h}_t$ | proposed replacement memory |

The final interpolation:

$$
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde{h}_t
$$

acts like a learned residual update over time.

## GRU vs LSTM

| Aspect | GRU | LSTM |
| --- | --- | --- |
| memory state | hidden state only | hidden plus cell state |
| gates | update and reset | input, forget, output |
| parameters | usually fewer | usually more |
| baseline role | compact recurrent baseline | stronger gated recurrent baseline |

The choice is rarely the main scientific claim; what matters is whether recurrent baselines are tuned fairly.

## Evaluation Boundary

| Claim | Check |
| --- | --- |
| efficient sequence model | parameter count, latency, sequence length |
| long-range modeling | truncation and hidden-state bottleneck |
| streaming model | state reset/carry policy |
| protein representation | homolog split and representation pooling |

## Practical Checks

- Check whether the task needs bidirectional context.
- Track how padding masks interact with hidden-state updates.
- Verify whether the final representation is last state, pooled states, or token-level outputs.
- Compare with [[concepts/architectures/state-space-model|state-space models]] when long sequences are central.
- Are GRU and LSTM baselines matched in hidden size, parameters, and tuning budget?
- Does padding update the hidden state accidentally?

## Related

- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/lstm|LSTM]]
- [[concepts/architectures/state-space-model|State-space models]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
