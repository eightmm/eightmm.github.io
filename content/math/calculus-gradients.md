---
title: Calculus and Gradients
aliases:
  - math/calculus-optimization
tags:
  - math
  - calculus
  - gradients
---

# Calculus and Gradients

미적분은 local change를 설명하는 수학입니다. AI에서는 loss가 input, parameter, coordinate, time에 대해 어떻게 변하는지를 말할 때 가장 자주 등장합니다.

The core object is the gradient of a scalar loss:

$$
\nabla_\theta \mathcal{L}(\theta)
=
\left[
\frac{\partial \mathcal{L}}{\partial \theta_1},
\frac{\partial \mathcal{L}}{\partial \theta_2},
\ldots,
\frac{\partial \mathcal{L}}{\partial \theta_d}
\right]^\top
$$

이 페이지는 Math-facing gateway입니다. derivative, chain rule, Jacobian, Hessian, backpropagation notation을 설명합니다. AdamW, warmup, clipping, batch-size scaling 같은 실전 optimizer 선택은 [[ai/machine-learning|Machine Learning]]과 [[concepts/machine-learning/optimization|Optimization]]에서 봅니다.

## Route Map

| Question | Start | Use For |
| --- | --- | --- |
| What is local change? | [Calculus](/concepts/math/calculus), [Chain rule](/concepts/math/chain-rule), [Matrix calculus](/concepts/math/matrix-calculus) | derivatives, gradients, composition, tensor notation |
| How do vector outputs change? | [Jacobian and Hessian](/concepts/math/jacobian-hessian) | sensitivity, curvature, normalizing flows, coordinate models |
| How does a network compute gradients? | [Backpropagation](/concepts/machine-learning/backpropagation), [Automatic differentiation](/concepts/machine-learning/automatic-differentiation) | training loops and differentiable programs |
| How do constraints enter? | [Constrained optimization](/concepts/math/constrained-optimization) | projected updates, constrained generation, feasibility |
| How do we debug gradient behavior? | [Gradient checking](/concepts/machine-learning/gradient-checking), [Loss landscape](/concepts/machine-learning/loss-landscape), [Second-order optimization](/concepts/machine-learning/second-order-optimization) | instability, curvature, implementation checks |

## Derivative

For a scalar function $f:\mathbb{R}\to\mathbb{R}$:

$$
\frac{df}{dx}
=
\lim_{\epsilon \to 0}
\frac{f(x+\epsilon)-f(x)}{\epsilon}
$$

The derivative measures the local slope. In training, the scalar function is often a loss and the variable is a model parameter.

## Gradient

For a scalar function $f:\mathbb{R}^d\to\mathbb{R}$:

$$
\nabla_x f(x)
=
\left[
\frac{\partial f}{\partial x_1},
\ldots,
\frac{\partial f}{\partial x_d}
\right]^\top
$$

The negative gradient gives the steepest local descent direction under the Euclidean norm:

$$
x_{t+1}
=
x_t - \eta \nabla_x f(x_t)
$$

This formula is the mathematical skeleton behind [[concepts/machine-learning/gradient-descent|Gradient descent]], but the engineering details belong to optimizer notes.

## Jacobian

For a vector-valued function $f:\mathbb{R}^n\to\mathbb{R}^m$:

$$
J_f(x)_{ij}
=
\frac{\partial f_i(x)}{\partial x_j}
$$

The Jacobian maps a small input perturbation to a first-order output perturbation:

$$
f(x+\delta)
\approx
f(x) + J_f(x)\delta
$$

Jacobians are useful for sensitivity analysis, coordinate prediction, normalizing flows, and geometric models.

## Hessian

For a scalar function $f:\mathbb{R}^d\to\mathbb{R}$:

$$
H_f(x)_{ij}
=
\frac{\partial^2 f(x)}{\partial x_i \partial x_j}
$$

The Hessian describes local curvature:

$$
f(x+\delta)
\approx
f(x)
+
\nabla f(x)^\top \delta
+
\frac{1}{2}\delta^\top H_f(x)\delta
$$

Large or ill-conditioned curvature can make training sensitive to learning rate, initialization, normalization, and numerical precision.

## Chain Rule

For a composition:

$$
y = f(h), \quad h = g(x)
$$

the derivative is:

$$
\frac{\partial y}{\partial x}
=
\frac{\partial y}{\partial h}
\frac{\partial h}{\partial x}
$$

Neural network backpropagation repeatedly applies this rule from the loss back to earlier layers and parameters:

$$
\frac{\partial \mathcal{L}}{\partial \theta}
=
\frac{\partial \mathcal{L}}{\partial h}
\frac{\partial h}{\partial \theta}
$$

For the canonical note, see [[concepts/math/chain-rule|Chain rule]].

## Matrix Calculus Patterns

For a linear layer:

$$
y = Wx + b
$$

and upstream gradient $\bar{y}=\partial \mathcal{L}/\partial y$:

$$
\frac{\partial \mathcal{L}}{\partial W}
=
\bar{y}x^\top,
\quad
\frac{\partial \mathcal{L}}{\partial x}
=
W^\top \bar{y},
\quad
\frac{\partial \mathcal{L}}{\partial b}
=
\bar{y}
$$

These shapes are often the fastest way to catch implementation mistakes.

## Boundary With AI Optimization

Keep these in Math:

- derivative definitions
- gradients, Jacobians, Hessians
- chain rule and matrix calculus
- curvature and local approximation
- second-order optimization diagnostics
- Lagrangians, KKT conditions, projection, and feasibility

Keep these in AI:

- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/adam|Adam]]
- [[concepts/machine-learning/adamw|AdamW]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/gradient-clipping|Gradient clipping]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/training-stability|Training stability]]

## Checks

- Is the derivative taken with respect to parameters, inputs, coordinates, or time?
- Is the object scalar, vector, matrix, sequence, graph, or coordinate set?
- Are row/column conventions and batch dimensions explicit?
- Does the loss have nondifferentiable points, masks, detach operations, or discrete sampling?
- Is the issue mathematical curvature, optimizer choice, numerical precision, or data scale?

## Related

- [[math/index|Math]]
- [[concepts/math/chain-rule|Chain rule]]
- [[math/numerical-computing|Numerical computing]]
- [[ai/machine-learning|Machine Learning]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/normalization|Normalization]]
