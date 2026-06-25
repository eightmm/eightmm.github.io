---
title: Gating
tags:
  - architectures
  - neural-networks
---

# Gating

Gating lets a model modulate information flow with learned multiplicative controls. It appears in RNNs, GLUs, state-space models, mixture-of-experts routers, and some attention variants.

A simple gate is:

$$
g = \sigma(xW_g + b_g)
$$

$$
y = g \odot f(x)
$$

where $g$ controls how much of $f(x)$ passes through.

## Where It Appears

- LSTM and GRU update gates in [[concepts/architectures/rnn|RNNs]].
- Selective updates in [[concepts/architectures/mamba|Mamba]] and other state-space models.
- Gated linear units in feed-forward blocks.
- Routing weights in [[concepts/architectures/mixture-of-experts|Mixture of experts]].

## Checks

- Is the gate scalar, channel-wise, token-wise, or expert-wise?
- Does the gate saturate near 0 or 1?
- Is gating used for memory, routing, feature modulation, or output control?
- Does the gate create an interpretation claim that needs verification?

## Related

- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/mixture-of-experts|Mixture of experts]]
