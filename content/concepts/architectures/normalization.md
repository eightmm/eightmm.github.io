---
title: Normalization
tags:
  - architectures
  - neural-networks
  - optimization
---

# Normalization

Normalization stabilizes training by rescaling activations or features. In modern sequence models, LayerNorm and RMSNorm are especially important.

Layer normalization computes statistics across the feature dimension:

$$
\mu = \frac{1}{d}\sum_{j=1}^{d} x_j,
\qquad
\sigma^2 = \frac{1}{d}\sum_{j=1}^{d}(x_j-\mu)^2
$$

$$
\operatorname{LayerNorm}(x)
= \gamma \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

RMSNorm removes mean subtraction:

$$
\operatorname{RMSNorm}(x)
= \gamma \frac{x}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\epsilon}}
$$

## Where It Appears

- Pre-norm and post-norm [[concepts/architectures/transformer|Transformer]] blocks.
- Stabilizing deep residual networks.
- Training large sequence models with long contexts.

## Checks

- Identify LayerNorm, BatchNorm, RMSNorm, or GroupNorm.
- Check whether normalization happens before or after residual branches.
- BatchNorm depends on batch statistics; LayerNorm does not.
- Normalization can change behavior between training and inference depending on type.

## Related

- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/machine-learning/optimization|Optimization]]
