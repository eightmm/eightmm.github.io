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

## Why It Matters

- Makes very deep networks easier to optimize.
- Lets layers learn refinements around an identity path.
- Connects directly to [[concepts/architectures/residual-connection|residual connections]] in Transformers and other models.

## Checks

- Does the residual branch preserve shape, or does it need a projection?
- Are normalization and activation placed before or after the residual addition?
- Is the residual path scaled, gated, or stochastic?

## Related

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/u-net|U-Net]]
