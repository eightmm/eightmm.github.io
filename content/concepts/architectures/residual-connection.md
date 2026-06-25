---
title: Residual Connection
tags:
  - architectures
  - neural-networks
---

# Residual Connection

A residual connection adds a module output back to its input. It helps deep networks preserve information and makes optimization easier.

$$
y = x + F(x)
$$

In a pre-norm Transformer block:

$$
x' = x + \operatorname{MHA}(\operatorname{LN}(x))
$$

$$
y = x' + \operatorname{FFN}(\operatorname{LN}(x'))
$$

Here $\operatorname{MHA}$ is multi-head attention, $\operatorname{LN}$ is layer normalization, and $\operatorname{FFN}$ is a token-wise [[concepts/architectures/feed-forward-network|feed-forward network]].

## Why It Matters

- Gradients can flow through the identity path.
- Deep models can learn small refinements instead of replacing representations at every layer.
- Residual paths require matching tensor shapes.

## Checks

- Confirm that dimensions match before addition.
- Check whether residual branches are scaled, gated, or dropped.
- Distinguish residual addition from concatenation.
- Watch hidden-state magnitude growth in very deep networks.

## Related

- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/mlp|MLP]]
