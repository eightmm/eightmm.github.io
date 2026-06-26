---
title: Numerical Computing
tags:
  - math
  - numerical-computing
  - stability
---

# Numerical Computing

Numerical computing is the math of finite-precision computation. AI formulas are written over real numbers, but training and inference run with floating-point tensors, limited memory, and finite accumulation order.

This page belongs in Math because it explains why algebraically equivalent formulas can behave differently on hardware. System-specific tradeoffs connect to [[infra/gpu/index|GPU infra]], [[infra/training/index|Training infra]], and [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]].

## Core Notes

- [[concepts/architectures/softmax|Softmax]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/gradient-clipping|Gradient clipping]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[infra/gpu/index|GPU infra]]
- [[infra/training/index|Training infra]]

## Floating Point

Floating-point numbers approximate real numbers with finite precision:

$$
\operatorname{fl}(x)
=
x(1+\delta),
\quad
|\delta| \le \epsilon
$$

Here $\epsilon$ is a machine-dependent precision scale. Smaller precision saves memory and bandwidth but increases rounding sensitivity.

## Overflow and Underflow

Exponentials can overflow for large positive inputs and underflow for large negative inputs:

$$
\exp(x)
\to
\infty
\quad
\text{or}
\quad
0
$$

This matters for softmax, likelihoods, attention, contrastive learning, and energy-based scoring.

## Log-Sum-Exp

The stable log-sum-exp trick subtracts the maximum before exponentiation:

$$
\operatorname{logsumexp}(x)
=
\log \sum_i \exp(x_i)
=
m + \log \sum_i \exp(x_i - m),
\quad
m = \max_i x_i
$$

This keeps the largest exponent at $\exp(0)=1$.

## Stable Softmax

Softmax should usually be computed as:

$$
\operatorname{softmax}(x)_i
=
\frac{\exp(x_i-m)}
{\sum_j \exp(x_j-m)},
\quad
m=\max_j x_j
$$

This is mathematically equivalent to ordinary softmax but numerically safer.

## Conditioning

A problem is ill-conditioned when small input changes can create large output changes. For a matrix $A$, the condition number is:

$$
\kappa(A)
=
\|A\| \|A^{-1}\|
$$

Large $\kappa(A)$ means solving systems, inverting matrices, or propagating gradients can be sensitive to noise and rounding.

## Precision in Training

Mixed precision changes memory and throughput, but it also changes numerical behavior. Common risk points:

- tiny gradients underflowing to zero
- large activations or logits overflowing
- reductions accumulating in a different order
- normalization statistics losing precision
- optimizer state requiring more precision than activations

Loss scaling is one way to reduce gradient underflow:

$$
\nabla_\theta (s\mathcal{L})
=
s\nabla_\theta \mathcal{L}
$$

The scaled gradient is later unscaled before the optimizer update.

## Reduction Order

Floating-point addition is not perfectly associative:

$$
(a+b)+c
\neq
a+(b+c)
$$

Parallel reductions, distributed training, and different kernels can therefore produce slightly different results even when the mathematical expression is the same.

## Checks

- Are logits, losses, and probabilities computed with stable formulas?
- Are reductions performed in a precision suitable for their dynamic range?
- Is instability caused by overflow, underflow, NaN propagation, or gradient explosion?
- Is the issue mathematical, numerical, architectural, optimizer-related, or hardware-related?
- Are deterministic expectations realistic for the kernels and distributed setup?

## Related

- [[math/index|Math]]
- [[math/calculus-gradients|Calculus and gradients]]
- [[concepts/architectures/softmax|Softmax]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
