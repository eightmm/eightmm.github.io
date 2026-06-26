---
title: Weight Initialization
tags:
  - architectures
  - optimization
  - neural-networks
---

# Weight Initialization

Weight initialization sets model parameters before training. It controls early activation scale, gradient scale, symmetry breaking, and whether optimization starts in a stable regime.

A common goal is to keep activation variance roughly constant across layers:

$$
\operatorname{Var}[h_{\ell+1}]
\approx
\operatorname{Var}[h_\ell]
$$

If the variance grows or shrinks exponentially with depth, gradients can explode or vanish before the optimizer has a chance to learn.

## Xavier / Glorot Initialization

For tanh-like activations or balanced fan-in/fan-out layers:

$$
W_{ij} \sim \mathcal{U}
\left(
-\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}},
\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}}
\right)
$$

$n_{\mathrm{in}}$ is fan-in and $n_{\mathrm{out}}$ is fan-out.

## Kaiming / He Initialization

For ReLU-like activations, a common variance target is:

$$
W_{ij} \sim \mathcal{N}
\left(
0,
\frac{2}{n_{\mathrm{in}}}
\right)
$$

The factor accounts for the ReLU discarding roughly half the signal.

## Modern Notes

- Residual branches may need scaling in very deep networks.
- Normalization reduces but does not remove initialization sensitivity.
- Embeddings, output heads, gates, and attention projections may use different initialization rules.
- Zero or near-zero initialization can be useful for residual adapters or final projection layers, but dangerous if it prevents symmetry breaking.

## Checks

- Is initialization matched to activation function and fan-in/fan-out?
- Are residual branches scaled deliberately?
- Are pretrained weights, random initialization, adapters, and task heads initialized differently?
- Does the run log the seed and model version?
- Does instability appear before data or optimizer effects can explain it?

## Related

- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
