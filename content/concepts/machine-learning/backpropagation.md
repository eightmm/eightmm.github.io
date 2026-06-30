---
title: Backpropagation
tags:
  - machine-learning
  - optimization
---

# Backpropagation

Backpropagation computes gradients through a composed computation graph using the chain rule. It is reverse-mode [[concepts/machine-learning/automatic-differentiation|Automatic differentiation]] applied to a scalar training loss.

For a composition:

$$
y = f(g(x))
$$

the chain rule is:

$$
\frac{\partial y}{\partial x}
=
\frac{\partial f}{\partial g}
\frac{\partial g}{\partial x}
$$

For model training:

$$
\nabla_\theta \mathcal{L}
=
\frac{\partial \mathcal{L}}{\partial f_\theta(x)}
\frac{\partial f_\theta(x)}{\partial \theta}
$$

More generally, each node $h_i$ in the graph receives an adjoint:

$$
\bar{h}_i
=
\frac{\partial \mathcal{L}}{\partial h_i}
$$

and sends it backward through local Jacobians:

$$
\bar{h}_{i-1}
=
\left(
\frac{\partial h_i}{\partial h_{i-1}}
\right)^\top
\bar{h}_i
$$

This is usually implemented as vector-Jacobian products rather than explicit Jacobian matrices.

## Reverse-Mode View

For an intermediate node:

$$
h_i = f_i(h_{p_1}, h_{p_2}, \ldots)
$$

the backward pass accumulates gradient contributions from all downstream users:

$$
\bar{h}_i
=
\sum_{j\in \operatorname{children}(i)}
\left(
\frac{\partial h_j}{\partial h_i}
\right)^\top
\bar{h}_j
$$

This accumulation is why reused tensors, shared parameters, and residual branches require careful gradient accounting.

For a parameter $\theta$ used in multiple places:

$$
\nabla_\theta \mathcal{L}
=
\sum_{u\in \operatorname{uses}(\theta)}
\left(
\frac{\partial h_u}{\partial \theta}
\right)^\top
\bar{h}_u
$$

The optimizer updates the parameter after gradients from the chosen mini-batch and accumulation steps have been combined.

## Loss Scaling

If the training loss is averaged over valid elements:

$$
\mathcal{L}
=
\frac{1}{|\Omega|}
\sum_{i\in\Omega}\ell_i
$$

then masked or padded elements must be removed before the reduction. Otherwise the gradient scale depends on padding length:

$$
\nabla_\theta \mathcal{L}
\propto
\frac{1}{N_{\mathrm{tokens}}}
\sum_i \nabla_\theta \ell_i
$$

instead of the intended valid-token count.

## What Can Go Wrong

- A tensor is detached or converted outside the differentiable graph.
- A mask is applied after loss reduction instead of before it.
- A custom operation has a backward rule that does not match the forward computation.
- Gradient accumulation unintentionally scales the effective loss.
- Shared parameters receive multiple gradient contributions that are not expected.
- In-place operations overwrite values needed for backward.
- Mixed-precision scaling hides overflow or underflow if skipped silently.

## Checks

- Are invalid or padded elements masked before loss reduction?
- Are gradients zeroed before the next step unless accumulation is intentional?
- Are frozen parameters excluded from gradient updates?
- Are gradient norms monitored when training is unstable?
- Does [[concepts/machine-learning/gradient-checking|Gradient checking]] pass for custom differentiable code?
- Does the reported loss reduction match the intended example/token/object unit?
- Are gradient accumulation, distributed averaging, and batch-size scaling documented?

## Related

- [[concepts/machine-learning/automatic-differentiation|Automatic differentiation]]
- [[concepts/machine-learning/gradient-checking|Gradient checking]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/gradient-clipping|Gradient clipping]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/math/jacobian-hessian|Jacobian and Hessian]]
- [[infra/gpu/index#memory|GPU memory]]
