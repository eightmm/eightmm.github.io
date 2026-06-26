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

## Gradient View

For a block:

$$
y = x + F_\theta(x)
$$

the Jacobian is:

$$
\frac{\partial y}{\partial x}
=
I + \frac{\partial F_\theta(x)}{\partial x}
$$

The identity term gives gradients a direct route through the network. This does not guarantee stability, but it reduces the need for every layer to preserve information through a nonlinear transformation.

## Shape Contract

Residual addition requires the branch output to match the residual stream:

$$
x\in\mathbb{R}^{B\times T\times d},
\qquad
F(x)\in\mathbb{R}^{B\times T\times d}
$$

If a block changes width, resolution, node count, or coordinate type, the skip path needs a projection:

$$
y = P(x) + F(x)
$$

where $P$ may be a linear layer, convolution, pooling, or graph-compatible projection.

## Variants

| Variant | Form | Use |
| --- | --- | --- |
| plain residual | $y=x+F(x)$ | standard residual block |
| scaled residual | $y=x+\alpha F(x)$ | control activation growth in deep networks |
| gated residual | $y=x+g(x)\odot F(x)$ | adaptive update strength |
| projected residual | $y=P(x)+F(x)$ | shape or channel changes |
| stochastic depth | randomly drop $F(x)$ during training | regularization for deep models |

In papers, changes to residual scaling or gating can be as important as the named architecture family.

## Checks

- Confirm that dimensions match before addition.
- Check whether residual branches are scaled, gated, or dropped.
- Distinguish residual addition from concatenation.
- Watch hidden-state magnitude growth in very deep networks.
- Check whether the residual stream represents tokens, nodes, pixels, residues, or coordinates.
- For equivariant models, confirm the residual path adds features of the same representation type.

## Related

- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/normalization-placement|Normalization placement]]
- [[concepts/architectures/weight-initialization|Weight initialization]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/mlp|MLP]]
