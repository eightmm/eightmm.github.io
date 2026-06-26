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

## What Can Go Wrong

- A tensor is detached or converted outside the differentiable graph.
- A mask is applied after loss reduction instead of before it.
- A custom operation has a backward rule that does not match the forward computation.
- Gradient accumulation unintentionally scales the effective loss.
- Shared parameters receive multiple gradient contributions that are not expected.

## Checks

- Are invalid or padded elements masked before loss reduction?
- Are gradients zeroed before the next step unless accumulation is intentional?
- Are frozen parameters excluded from gradient updates?
- Are gradient norms monitored when training is unstable?
- Does [[concepts/machine-learning/gradient-checking|Gradient checking]] pass for custom differentiable code?

## Related

- [[concepts/machine-learning/automatic-differentiation|Automatic differentiation]]
- [[concepts/machine-learning/gradient-checking|Gradient checking]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/math/jacobian-hessian|Jacobian and Hessian]]
- [[infra/gpu/index#memory|GPU memory]]
