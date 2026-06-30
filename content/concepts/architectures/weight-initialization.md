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

For a linear layer

$$
h = Wx
$$

with independent weights and inputs, a rough variance propagation argument gives:

$$
\operatorname{Var}[h_j]
\approx
n_{\mathrm{in}}\operatorname{Var}[W_{ji}]\operatorname{Var}[x_i]
$$

So keeping variance stable suggests:

$$
\operatorname{Var}[W]
\propto
\frac{1}{n_{\mathrm{in}}}
$$

The exact constant depends on activation function, fan-out, residual path, and normalization.

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

## Fan-In and Fan-Out

For a dense matrix $W\in\mathbb{R}^{n_{\mathrm{out}}\times n_{\mathrm{in}}}$:

$$
\operatorname{fan\_in}=n_{\mathrm{in}},
\quad
\operatorname{fan\_out}=n_{\mathrm{out}}
$$

For convolution, fan-in and fan-out include kernel area:

$$
\operatorname{fan\_in}
=
C_{\mathrm{in}}\prod_d K_d,
\quad
\operatorname{fan\_out}
=
C_{\mathrm{out}}\prod_d K_d
$$

| Initialization | Typical use | Variance scale |
| --- | --- | --- |
| Xavier / Glorot | tanh, sigmoid, balanced forward/backward flow | $\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}$ |
| Kaiming / He | ReLU-family activations | $\frac{2}{n_{\mathrm{in}}}$ |
| orthogonal | recurrent or deep linear-like blocks | preserves norm under assumptions |
| small final layer | residual adapters, output heads | starts near identity/no-op |
| zero bias | common default | avoids initial offset |

## Residual and Normalized Networks

Residual networks change the initialization question because each block adds to an existing stream:

$$
h_{\ell+1}
=
h_\ell + F_\ell(h_\ell)
$$

If $F_\ell$ is too large at initialization, the residual stream scale can grow with depth. Common stabilizers include smaller residual branch initialization, normalization placement, and zero-initialized final projections in residual blocks.

| Component | Initialization concern |
| --- | --- |
| embedding table | scale affects early logits and representation norm |
| attention projections | query/key scale affects softmax sharpness |
| gate bias | controls initially open/closed memory path |
| residual branch | should not overwhelm skip path at start |
| output head | affects initial loss scale and calibration |

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
- Are fan-in/fan-out computed correctly for convolution or grouped layers?
- Are gate biases and final projections initialized intentionally?

## Related

- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
