---
title: Normalization Placement
tags:
  - architectures
  - normalization
  - transformer
---

# Normalization Placement

Normalization placement describes where normalization is applied relative to residual branches. The placement can strongly affect gradient flow, depth scaling, and training stability.

For a residual block with sublayer $F$:

$$
\text{pre-norm:}\quad
y = x + F(\operatorname{Norm}(x))
$$

$$
\text{post-norm:}\quad
y = \operatorname{Norm}(x + F(x))
$$

Pre-norm keeps a direct identity path from output to input, while post-norm normalizes the residual sum.

## Transformer Block

A pre-norm Transformer block is often written as:

$$
x' = x + \operatorname{MHA}(\operatorname{Norm}(x))
$$

$$
y = x' + \operatorname{FFN}(\operatorname{Norm}(x'))
$$

The normalization sits before attention and feed-forward sublayers.

## Tradeoffs

- Pre-norm usually improves optimization stability for deep Transformers.
- Post-norm can make layer outputs more uniformly normalized but can be harder to optimize at depth.
- Sandwich or hybrid norms add normalization both before and after sublayers.
- Residual scaling, initialization, and learning-rate schedule interact with placement.

## Checks

- Is the block pre-norm, post-norm, sandwich-norm, or another variant?
- Is the normalization LayerNorm, RMSNorm, BatchNorm, or GroupNorm?
- Are residual branches scaled or gated?
- Is the reported architecture comparable to the baseline after changing norm placement?
- Does instability appear as depth increases?

## Related

- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/weight-initialization|Weight initialization]]
- [[concepts/machine-learning/training-stability|Training stability]]
