---
title: Linear Layer
tags:
  - architectures
  - neural-networks
---

# Linear Layer

A linear layer maps each input vector to an output vector with an affine transform. It is the basic learnable block inside MLPs, attention projections, [[concepts/architectures/feed-forward-network|feed-forward networks]], classifiers, and adapters.

$$
y = xW + b
$$

For a batch or sequence, $x\in\mathbb{R}^{T\times d_{\mathrm{in}}}$, $W\in\mathbb{R}^{d_{\mathrm{in}}\times d_{\mathrm{out}}}$, $b\in\mathbb{R}^{d_{\mathrm{out}}}$, and $y\in\mathbb{R}^{T\times d_{\mathrm{out}}}$.

For a batch of sequences:

$$
X \in \mathbb{R}^{B \times T \times d_{\mathrm{in}}}
$$

the same matrix is applied independently at each batch and token position:

$$
Y_{b,t,:}
=
X_{b,t,:}W + b
$$

This is parameter sharing over positions. The layer does not mix tokens by itself; it only mixes channels/features within each position. Token mixing comes from [[concepts/architectures/attention|Attention]], convolution, recurrence, pooling, or message passing.

## Parameter Count

With bias, the number of parameters is:

$$
d_{\mathrm{in}}d_{\mathrm{out}}
+
d_{\mathrm{out}}
$$

Without bias:

$$
d_{\mathrm{in}}d_{\mathrm{out}}
$$

This count matters when comparing projection-heavy architectures. A Transformer block often contains several large linear projections: query, key, value, output projection, and feed-forward projections.

## Backward View

For $Y=XW+b$ and upstream gradient $G=\partial \mathcal{L}/\partial Y$:

$$
\frac{\partial \mathcal{L}}{\partial W}
=
X^\top G
$$

$$
\frac{\partial \mathcal{L}}{\partial b}
=
\sum_i G_i
$$

$$
\frac{\partial \mathcal{L}}{\partial X}
=
GW^\top
$$

This is one reason linear layers are easy to optimize but easy to overuse: they are expressive in channel space, but they do not add locality, order, or graph structure by themselves.

## Design Choices

- Bias: often removed before normalization or in some attention projections.
- Width: controls representation capacity and memory.
- Low-rank factorization: replaces $W$ with $AB$ to reduce parameters.
- Weight tying: reuses a matrix across input embedding and output projection.
- Initialization: affects activation scale and gradient flow in deep networks.

## Where It Appears

- Query, key, and value projections in [[concepts/architectures/attention|Attention]].
- Feed-forward blocks in [[concepts/architectures/transformer|Transformer]].
- Projection heads in [[concepts/learning/contrastive-learning|Contrastive learning]].
- Readout layers in [[concepts/architectures/gnn|Graph neural networks]].
- Low-rank adapters and task-specific heads in [[concepts/learning/fine-tuning|Fine-tuning]].

## Checks

- Track whether tensors are row-major or column-major in the implementation.
- Confirm whether the layer includes a bias term.
- Watch dimension changes at residual connections.
- Distinguish channel mixing from token, spatial, graph, or structure mixing.
- Check parameter count when comparing architecture variants.
- Distinguish a linear layer in a neural network from a statistical [[concepts/machine-learning/linear-model|Linear model]].

## Related

- [[concepts/architectures/mlp|MLP]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/parameter-sharing|Parameter sharing]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
