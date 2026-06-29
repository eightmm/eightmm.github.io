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

핵심 object는 scalar loss의 gradient입니다.

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

| Question | Start | Use for |
| --- | --- | --- |
| local change란 무엇인가? | [Calculus](/concepts/math/calculus), [Chain rule](/concepts/math/chain-rule), [Matrix calculus](/concepts/math/matrix-calculus) | derivative, gradient, composition, tensor notation |
| vector output은 어떻게 변하는가? | [Jacobian and Hessian](/concepts/math/jacobian-hessian) | sensitivity, curvature, normalizing flow, coordinate model |
| network는 gradient를 어떻게 계산하는가? | [Backpropagation](/concepts/machine-learning/backpropagation), [Automatic differentiation](/concepts/machine-learning/automatic-differentiation) | training loop와 differentiable program |
| constraint는 어떻게 들어가는가? | [Constrained optimization](/concepts/math/constrained-optimization) | projected update, constrained generation, feasibility |
| gradient behavior는 어떻게 debug하는가? | [Gradient checking](/concepts/machine-learning/gradient-checking), [Loss landscape](/concepts/machine-learning/loss-landscape), [Second-order optimization](/concepts/machine-learning/second-order-optimization) | instability, curvature, implementation check |

## Derivative

Scalar function $f:\mathbb{R}\to\mathbb{R}$에 대해:

$$
\frac{df}{dx}
=
\lim_{\epsilon \to 0}
\frac{f(x+\epsilon)-f(x)}{\epsilon}
$$

Derivative는 local slope를 측정합니다. Training에서는 scalar function이 보통 loss이고, variable은 model parameter입니다.

## Gradient

Scalar function $f:\mathbb{R}^d\to\mathbb{R}$에 대해:

$$
\nabla_x f(x)
=
\left[
\frac{\partial f}{\partial x_1},
\ldots,
\frac{\partial f}{\partial x_d}
\right]^\top
$$

Negative gradient는 Euclidean norm 기준 가장 가파른 local descent direction을 줍니다.

$$
x_{t+1}
=
x_t - \eta \nabla_x f(x_t)
$$

이 식은 [[concepts/machine-learning/gradient-descent|Gradient descent]]의 수학적 뼈대입니다. 다만 engineering detail은 optimizer note에 둡니다.

## Jacobian

Vector-valued function $f:\mathbb{R}^n\to\mathbb{R}^m$에 대해:

$$
J_f(x)_{ij}
=
\frac{\partial f_i(x)}{\partial x_j}
$$

Jacobian은 작은 input perturbation을 first-order output perturbation으로 보냅니다.

$$
f(x+\delta)
\approx
f(x) + J_f(x)\delta
$$

Jacobian은 sensitivity analysis, coordinate prediction, normalizing flow, geometric model에서 유용합니다.

## Hessian

Scalar function $f:\mathbb{R}^d\to\mathbb{R}$에 대해:

$$
H_f(x)_{ij}
=
\frac{\partial^2 f(x)}{\partial x_i \partial x_j}
$$

Hessian은 local curvature를 설명합니다.

$$
f(x+\delta)
\approx
f(x)
+
\nabla f(x)^\top \delta
+
\frac{1}{2}\delta^\top H_f(x)\delta
$$

크거나 ill-conditioned인 curvature는 training을 learning rate, initialization, normalization, numerical precision에 민감하게 만듭니다.

## Chain Rule

Composition이 아래와 같을 때:

$$
y = f(h), \quad h = g(x)
$$

derivative는 아래와 같습니다.

$$
\frac{\partial y}{\partial x}
=
\frac{\partial y}{\partial h}
\frac{\partial h}{\partial x}
$$

Neural network backpropagation은 loss에서 earlier layer와 parameter로 거슬러 올라가며 이 rule을 반복 적용합니다.

$$
\frac{\partial \mathcal{L}}{\partial \theta}
=
\frac{\partial \mathcal{L}}{\partial h}
\frac{\partial h}{\partial \theta}
$$

Canonical note는 [[concepts/math/chain-rule|Chain rule]]에서 봅니다.

## Matrix Calculus Patterns

Linear layer가 아래와 같고:

$$
y = Wx + b
$$

위쪽에서 전달된 gradient가 $\bar{y}=\partial \mathcal{L}/\partial y$이면:

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

이 shape들은 implementation mistake를 잡는 가장 빠른 단서가 됩니다.

## Boundary With AI Optimization

Math에는 아래 내용을 둡니다.

- derivative definitions
- gradients, Jacobians, Hessians
- chain rule and matrix calculus
- curvature and local approximation
- second-order optimization diagnostics
- Lagrangians, KKT conditions, projection, and feasibility

AI에는 아래 내용을 둡니다.

- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/adam|Adam]]
- [[concepts/machine-learning/adamw|AdamW]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/gradient-clipping|Gradient clipping]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/training-stability|Training stability]]

## Checks

- derivative가 parameter, input, coordinate, time 중 무엇에 대해 계산되는가?
- object가 scalar, vector, matrix, sequence, graph, coordinate set 중 무엇인가?
- row/column convention과 batch dimension이 명시되어 있는가?
- loss에 nondifferentiable point, mask, detach operation, discrete sampling이 있는가?
- 문제가 mathematical curvature, optimizer choice, numerical precision, data scale 중 어디에 있는가?

## Related

- [[math/index|Math]]
- [[concepts/math/chain-rule|Chain rule]]
- [[math/numerical-computing|Numerical computing]]
- [[ai/machine-learning|Machine Learning]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/normalization|Normalization]]
