---
title: Normalization
tags:
  - architectures
  - neural-networks
  - optimization
---

# Normalization

Normalization stabilizes training by rescaling activations or features. In modern sequence models, LayerNorm and RMSNorm are especially important.

The general pattern is:

$$
\operatorname{Norm}(x)
=
\gamma \odot
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
+ \beta
$$

where $\mu$ and $\sigma^2$ are computed over a chosen axis, $\gamma$ and $\beta$ are learned scale and shift parameters, and $\epsilon$ prevents division by zero.

## LayerNorm

Layer normalization computes statistics across the feature dimension for each token or example:

$$
\mu = \frac{1}{d}\sum_{j=1}^{d} x_j,
\qquad
\sigma^2 = \frac{1}{d}\sum_{j=1}^{d}(x_j-\mu)^2
$$

$$
\operatorname{LayerNorm}(x)
= \gamma \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

For a token matrix $X\in\mathbb{R}^{T\times d}$, LayerNorm usually normalizes each row independently. This makes it common in [[concepts/architectures/transformer|Transformer]] blocks where batch size or sequence length can vary.

## RMSNorm

RMSNorm removes mean subtraction:

$$
\operatorname{RMSNorm}(x)
= \gamma \frac{x}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\epsilon}}
$$

RMSNorm keeps the direction of $x$ closer to the original vector and reduces computation slightly. Many modern large sequence models use RMSNorm instead of LayerNorm.

## BatchNorm

Batch normalization computes statistics over the batch and often spatial dimensions:

$$
\mu_B
=
\frac{1}{m}\sum_{i=1}^{m} x_i,
\qquad
\sigma_B^2
=
\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2
$$

$$
\operatorname{BatchNorm}(x_i)
=
\gamma\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}+\beta
$$

During inference, BatchNorm usually uses running estimates rather than current-batch statistics. This train/inference difference is why BatchNorm can be fragile with tiny batches, distribution shift, or gradient accumulation.

## Pre-Norm vs Post-Norm

For a residual block $F$:

$$
\text{pre-norm:}\quad
y=x+F(\operatorname{Norm}(x))
$$

$$
\text{post-norm:}\quad
y=\operatorname{Norm}(x+F(x))
$$

Pre-norm often improves optimization stability in deep Transformers because gradients can flow through the residual path more directly.

## Where It Appears

- Pre-norm and post-norm [[concepts/architectures/transformer|Transformer]] blocks.
- Stabilizing deep residual networks.
- Training large sequence models with long contexts.
- CNNs and vision models through BatchNorm, GroupNorm, or LayerNorm variants.

## Checks

- Identify LayerNorm, BatchNorm, RMSNorm, or GroupNorm.
- Check whether normalization happens before or after residual branches.
- BatchNorm depends on batch statistics; LayerNorm does not.
- Normalization can change behavior between training and inference depending on type.
- Check the normalized axis: feature, channel, batch, group, token, or spatial dimension.
- Check whether $\gamma$ and $\beta$ are present and whether bias is disabled in adjacent layers.

## Related

- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/optimization|Optimization]]
