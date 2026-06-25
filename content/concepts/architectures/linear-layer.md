---
title: Linear Layer
tags:
  - architectures
  - neural-networks
---

# Linear Layer

A linear layer maps each input vector to an output vector with an affine transform. It is the basic learnable block inside MLPs, attention projections, feed-forward networks, classifiers, and adapters.

$$
y = xW + b
$$

For a batch or sequence, $x\in\mathbb{R}^{T\times d_{\mathrm{in}}}$, $W\in\mathbb{R}^{d_{\mathrm{in}}\times d_{\mathrm{out}}}$, $b\in\mathbb{R}^{d_{\mathrm{out}}}$, and $y\in\mathbb{R}^{T\times d_{\mathrm{out}}}$.

## Where It Appears

- Query, key, and value projections in [[concepts/architectures/attention|Attention]].
- Feed-forward blocks in [[concepts/architectures/transformer|Transformer]].
- Projection heads in [[concepts/learning/contrastive-learning|Contrastive learning]].
- Readout layers in [[concepts/architectures/gnn|Graph neural networks]].

## Checks

- Track whether tensors are row-major or column-major in the implementation.
- Confirm whether the layer includes a bias term.
- Watch dimension changes at residual connections.
- Distinguish a linear layer in a neural network from a statistical [[concepts/machine-learning/linear-model|Linear model]].

## Related

- [[concepts/architectures/mlp|MLP]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/normalization|Normalization]]
