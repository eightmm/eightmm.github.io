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

where $g_t$ controls whether the state changes or is preserved. In GLU-style feed-forward blocks such as [[papers/architectures/glu-variants-improve-transformer|GLU Variants Improve Transformer]]:

$$
\operatorname{GLU}(x)
=
(xW_a)\odot\sigma(xW_b)
$$

The multiplicative path gives the model a data-dependent way to suppress, pass, or route features.

## Gate Types

| Gate type | Formula sketch | Appears in |
| --- | --- | --- |
| feature gate | $y=g(x)\odot f(x)$ | GLU, gated MLP |
| state update gate | $h' = g\odot \tilde{h} + (1-g)\odot h$ | GRU, LSTM-like memory |
| routing gate | $p(e\mid x)=\operatorname{softmax}(W_rx)$ | mixture-of-experts |
| selective scan gate | input-dependent transition or mixing | state-space models |
| attention gate | content-dependent weighted aggregation | attention variants |

The shared idea is multiplicative control. The difference is what is being controlled: feature magnitude, memory update, expert choice, or sequence dynamics.

## Gradient Behavior

For sigmoid gates:

$$
\sigma'(a)
=
\sigma(a)(1-\sigma(a))
$$

When $a$ is very positive or negative, $\sigma'(a)$ becomes small. Saturated gates can preserve or block information, but they also reduce gradient signal through the gate.

| Regime | Behavior | Risk |
| --- | --- | --- |
| $g\approx 0$ | suppress path | useful feature blocked |
| $g\approx 1$ | pass path | gate stops modulating |
| $g\approx 0.5$ | high sensitivity | unstable if scale is poorly normalized |
| sparse routing | few paths active | load imbalance or expert collapse |

## Gating vs Attention

Gating modulates a representation, while attention chooses or mixes information from a set of positions.

$$
\text{gate}: y_i = g_i \odot x_i
$$

$$
\text{attention}: y_i = \sum_j \alpha_{ij} v_j
$$

Both can be content-dependent, but they answer different questions:

| Mechanism | Question |
| --- | --- |
| gating | should this feature/state/path pass? |
| attention | which other positions should this position read? |
| routing | which expert/path should process this token? |

## Failure Modes

- Saturation: gates stay near 0 or 1 and reduce gradient signal.
- Routing collapse: a small set of experts or paths receives most tokens.
- Spurious interpretability: gate value is treated as explanation without intervention or ablation.
- Instability: multiplicative interactions amplify activation scale when normalization is weak.
- Confused explanation: a large gate value is not automatically causal evidence.

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
- Are gate logits normalized or regularized enough to avoid collapse?
- Is the gate measured by distribution, ablation, or only visual inspection?

## Related

- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/gru|GRU]]
- [[concepts/architectures/lstm|LSTM]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/mixture-of-experts|Mixture of experts]]
- [[concepts/architectures/normalization|Normalization]]
