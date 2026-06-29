---
title: Jacobian and Hessian
tags:
  - math
  - calculus
  - optimization
---

# Jacobian and Hessian

The Jacobian and Hessian describe first-order and second-order local behavior. They are central to [[concepts/math/matrix-calculus|Matrix calculus]], [[concepts/machine-learning/backpropagation|Backpropagation]], optimization, sensitivity analysis, normalizing flows, and geometric models.

For a vector-valued function $f:\mathbb{R}^d\to\mathbb{R}^m$, the Jacobian is:

$$
J_f(x)
=
\frac{\partial f}{\partial x}
=
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_d} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_d}
\end{bmatrix}
$$

For a scalar function $\mathcal{L}:\mathbb{R}^d\to\mathbb{R}$, the Hessian is:

$$
H_{\mathcal{L}}(x)
=
\nabla_x^2 \mathcal{L}(x)
=
\begin{bmatrix}
\frac{\partial^2 \mathcal{L}}{\partial x_1\partial x_1} & \cdots & \frac{\partial^2 \mathcal{L}}{\partial x_1\partial x_d} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 \mathcal{L}}{\partial x_d\partial x_1} & \cdots & \frac{\partial^2 \mathcal{L}}{\partial x_d\partial x_d}
\end{bmatrix}
$$

The gradient gives a local linear approximation:

$$
\mathcal{L}(x+\Delta x)
\approx
\mathcal{L}(x)
+
\nabla_x\mathcal{L}(x)^\top \Delta x
$$

The Hessian adds curvature:

$$
\mathcal{L}(x+\Delta x)
\approx
\mathcal{L}(x)
+
\nabla_x\mathcal{L}(x)^\top \Delta x
+
\frac{1}{2}
\Delta x^\top
H_{\mathcal{L}}(x)
\Delta x
$$

## Products Instead of Matrices

Deep learning systems rarely materialize full Jacobians or Hessians. They usually compute products:

$$
v^\top J_f(x)
$$

or

$$
H_{\mathcal{L}}(x)v
$$

Vector-Jacobian products are the workhorse of reverse-mode automatic differentiation. Hessian-vector products are useful for curvature diagnostics, second-order methods, and stability analysis.

## Local Sensitivity

The Jacobian maps a small input change to an output change:

$$
f(x+\Delta x)-f(x)
\approx
J_f(x)\Delta x
$$

This matters for robustness, attribution, normalizing flows, and stability. A large Jacobian norm can indicate high local sensitivity:

$$
\lVert J_f(x)\rVert
$$

but the chosen norm and input scaling must be stated.

## Curvature and Optimization

For a scalar loss, Hessian eigenvalues describe local curvature:

$$
H v_i = \lambda_i v_i
$$

| Curvature Pattern | Interpretation |
| --- | --- |
| $\lambda_i > 0$ | locally convex direction |
| $\lambda_i < 0$ | saddle or locally concave direction |
| large $\lvert\lambda_i\rvert$ | sharp or unstable direction |
| near-zero $\lambda_i$ | flat or underdetermined direction |

Gradient descent is locally stable in a quadratic bowl when the step size is small relative to the largest curvature:

$$
0 < \eta < \frac{2}{\lambda_{\max}}
$$

This is an intuition, not a full deep-network guarantee.

## Generative Model Uses

| Model Family | Jacobian/Hessian Role |
| --- | --- |
| normalizing flow | log determinant of Jacobian |
| continuous normalizing flow | divergence or trace of Jacobian |
| score model | gradient of log density $\nabla_x \log p(x)$ |
| energy model | force or update from $-\nabla_X E(X)$ |
| implicit layer | Jacobian inverse or fixed-point sensitivity |
| second-order optimizer | Hessian or Hessian approximation |

For coordinate-based molecular models, the derivative target can be coordinates rather than parameters.

## Reporting Contract

| Field | Question |
| --- | --- |
| Function | What function is differentiated? |
| Variable | With respect to input, parameter, coordinate, time, or latent? |
| Product | Full matrix, VJP, JVP, HVP, trace estimate, or determinant? |
| Approximation | Exact autograd, finite difference, stochastic estimator, or structured formula? |
| Cost | How many backward/forward passes are required? |

## Why It Matters

- Backpropagation computes vector-Jacobian products efficiently.
- Normalizing flows need log-determinants of Jacobians or structured alternatives.
- Optimization stability depends on curvature and Hessian eigenvalues.
- Coordinate-based models may need gradients with respect to positions as well as parameters.
- Sensitivity analysis asks how outputs change when inputs, coordinates, or parameters move.

## Checks

- Is the function scalar-valued or vector-valued?
- Is the derivative with respect to inputs, parameters, coordinates, or time?
- Does the method need a full Jacobian, a vector-Jacobian product, a Jacobian-vector product, or a Hessian-vector product?
- Is curvature being used for optimization, diagnostics, uncertainty, or stability?
- Are gradients intentionally stopped through any part of the computation?
- Are norms, traces, determinants, or eigenvalues estimated with enough numerical detail?
- Is the derivative evaluated on training data, validation data, generated samples, or OOD inputs?

## Related

- [[concepts/math/calculus|Calculus]]
- [[concepts/math/matrix-calculus|Matrix calculus]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/geometric-deep-learning/coordinate-update|Coordinate update]]
