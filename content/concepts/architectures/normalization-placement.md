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

## Gradient Path

In a pre-norm block:

$$
y = x + F(\operatorname{Norm}(x))
$$

the residual stream keeps an unnormalized identity path. In a post-norm block:

$$
y = \operatorname{Norm}(x + F(x))
$$

the gradient passes through normalization after the residual sum. This can affect depth scaling because normalization changes both the forward activation and the backward Jacobian.

## Common Patterns

| Pattern | Form | Typical Reading |
| --- | --- | --- |
| pre-norm | $x+F(\operatorname{Norm}(x))$ | stable for deep sequence models |
| post-norm | $\operatorname{Norm}(x+F(x))$ | output is normalized after each block |
| sandwich norm | $\operatorname{Norm}(x+F(\operatorname{Norm}(x)))$ | extra stabilization, extra compute |
| parallel block | $x+F_1(\operatorname{Norm}(x))+F_2(\operatorname{Norm}(x))$ | attention and FFN share the same residual input |
| norm-free or scaled | residual scaling without explicit norm | relies on initialization and scale control |

## Comparison Boundary

When comparing architectures, normalization placement is not a minor implementation detail. It changes:

- trainable depth;
- learning-rate tolerance;
- activation scale;
- compatibility with residual scaling;
- whether the baseline has the same stability budget.

If a paper changes both architecture family and norm placement, do not attribute all gains to the family name.

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
- Is the output before the task head normalized, or only the internal residual stream?
- Are inference-time normalization statistics identical to training-time behavior?

## Related

- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/weight-initialization|Weight initialization]]
- [[concepts/machine-learning/training-stability|Training stability]]
