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

Gated residual updates often have the form:

$$
h_{t+1}
=
g_t \odot \tilde{h}_{t+1}
+ (1-g_t)\odot h_t
$$

where $g_t$ controls whether the state changes or is preserved. In GLU-style feed-forward blocks:

$$
\operatorname{GLU}(x)
=
(xW_a)\odot\sigma(xW_b)
$$

The multiplicative path gives the model a data-dependent way to suppress, pass, or route features.

## Failure Modes

- Saturation: gates stay near 0 or 1 and reduce gradient signal.
- Routing collapse: a small set of experts or paths receives most tokens.
- Spurious interpretability: gate value is treated as explanation without intervention or ablation.
- Instability: multiplicative interactions amplify activation scale when normalization is weak.

## Where It Appears

- LSTM and GRU update gates in [[concepts/architectures/rnn|RNNs]].
- Selective updates in [[concepts/architectures/mamba|Mamba]] and other state-space models.
- Gated linear units in [[concepts/architectures/feed-forward-network|feed-forward networks]].
- Routing weights in [[concepts/architectures/mixture-of-experts|Mixture of experts]].

## Checks

- Is the gate scalar, channel-wise, token-wise, or expert-wise?
- Does the gate saturate near 0 or 1?
- Is gating used for memory, routing, feature modulation, or output control?
- Does the gate create an interpretation claim that needs verification?

## Related

- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/mixture-of-experts|Mixture of experts]]
- [[concepts/architectures/normalization|Normalization]]
