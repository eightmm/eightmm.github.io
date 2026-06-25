---
title: Residual Network
tags:
  - architectures
  - cnn
  - residual
---

# Residual Network

A residual network is a deep architecture built from residual blocks. It was introduced for vision models, but the residual pattern is now a general design principle across CNNs, Transformers, GNNs, and diffusion backbones.

The core block is:

$$
y = x + F_\theta(x)
$$

where $F_\theta$ is usually a small stack of convolutions, normalizations, and activations.

For a sequence of residual blocks:

$$
x_{l+1}
=
x_l + F_l(x_l)
$$

the representation after $L$ layers is:

$$
x_L
=
x_0
+
\sum_{l=0}^{L-1}
F_l(x_l)
$$

This additive path makes the network learn refinements around an identity signal rather than replacing the whole representation at every layer.

## Why It Matters

- Makes very deep networks easier to optimize.
- Lets layers learn refinements around an identity path.
- Connects directly to [[concepts/architectures/residual-connection|residual connections]] in Transformers and other models.

## Block Variants

Basic image residual block:

$$
y
=
x
+
\operatorname{Conv}_2(
\sigma(
\operatorname{Norm}(
\operatorname{Conv}_1(x)
)))
$$

Bottleneck block:

$$
F(x)
=
\operatorname{Conv}_{1\times1}^{\downarrow}
\rightarrow
\operatorname{Conv}_{3\times3}
\rightarrow
\operatorname{Conv}_{1\times1}^{\uparrow}
$$

The bottleneck reduces channel width before the expensive spatial convolution, then restores it.

When shape changes, the skip path uses a projection:

$$
y
=
P(x) + F_\theta(x)
$$

where $P$ may be a $1\times1$ convolution, linear projection, or pooling/projection pair.

## Residual Scaling

Some very deep networks scale the residual branch:

$$
y
=
x + \alpha F_\theta(x)
$$

with $\alpha < 1$ or learned gating. This controls hidden-state magnitude and improves stability in deep stacks.

## Where It Appears

- CNN backbones for image and grid inputs.
- [[concepts/architectures/u-net|U-Net]] encoder and decoder blocks.
- Transformer blocks through [[concepts/architectures/residual-connection|Residual connection]].
- Graph and geometric models that repeatedly refine node or coordinate states.

## Checks

- Does the residual branch preserve shape, or does it need a projection?
- Are normalization and activation placed before or after the residual addition?
- Is the residual path scaled, gated, or stochastic?
- Is the skip path identity, projection, downsampling, or cross-stage connection?
- Are residual blocks being compared at equal depth, width, and compute?

## Related

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/u-net|U-Net]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/machine-learning/training-stability|Training stability]]
