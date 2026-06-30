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

The residual stream is the running representation that every block reads from and writes back to. The branch $F$ is an update, not a replacement:

$$
\Delta x = F(x),
\qquad
y=x+\Delta x
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

For a deep stack:

$$
x_L
=
x_0
+
\sum_{\ell=0}^{L-1}
F_\ell(x_\ell)
$$

so residual networks can be read as iterative refinement of a shared representation.

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

For coordinate or equivariant features, residual addition is only valid between objects with the same transformation type. A scalar invariant feature and a vector equivariant feature should not be added directly.

## Variants

| Variant | Form | Use |
| --- | --- | --- |
| plain residual | $y=x+F(x)$ | standard residual block |
| scaled residual | $y=x+\alpha F(x)$ | control activation growth in deep networks |
| gated residual | $y=x+g(x)\odot F(x)$ | adaptive update strength |
| projected residual | $y=P(x)+F(x)$ | shape or channel changes |
| stochastic depth | randomly drop $F(x)$ during training | regularization for deep models |

In papers, changes to residual scaling or gating can be as important as the named architecture family.

## Stability Boundary

Residual branches can still grow too large. Deep models often combine residuals with:

| Stabilizer | Role |
| --- | --- |
| normalization | controls branch input or output scale |
| residual scaling | reduces update magnitude |
| careful initialization | starts blocks near identity |
| dropout or stochastic depth | regularizes branch updates |
| gradient clipping | handles occasional large updates |

For scaled residuals:

$$
y=x+\alpha F(x)
$$

small $\alpha$ keeps early updates close to identity. Some architectures learn or schedule $\alpha$.

## Checks

- Confirm that dimensions match before addition.
- Check whether residual branches are scaled, gated, or dropped.
- Distinguish residual addition from concatenation.
- Watch hidden-state magnitude growth in very deep networks.
- Check whether the residual stream represents tokens, nodes, pixels, residues, or coordinates.
- For equivariant models, confirm the residual path adds features of the same representation type.
- Check whether residual scaling, gating, or stochastic depth changes between train and inference.
- Check whether projection shortcuts change the claim by mixing or downsampling the input.

## Related

- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/normalization-placement|Normalization placement]]
- [[concepts/architectures/weight-initialization|Weight initialization]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/mlp|MLP]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]]
