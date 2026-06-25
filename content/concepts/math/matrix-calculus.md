---
title: Matrix Calculus
tags:
  - math
  - matrix-calculus
  - optimization
---

# Matrix Calculus

Matrix calculus extends derivatives to vectors, matrices, and tensor-valued computations. It is the notation behind gradients, Jacobians, neural network layers, and coordinate updates.

For a function $f:\mathbb{R}^d\to\mathbb{R}^m$, the Jacobian is:

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

For a linear map:

$$
y = Wx + b
$$

the Jacobian with respect to $x$ is:

$$
\frac{\partial y}{\partial x}=W
$$

For a scalar loss $\mathcal{L}$ and parameters $\theta$, optimization uses:

$$
\nabla_\theta \mathcal{L}
=
\left[
\frac{\partial \mathcal{L}}{\partial \theta_1},
\ldots,
\frac{\partial \mathcal{L}}{\partial \theta_n}
\right]^\top
$$

## Key Ideas

- Gradients are for scalar outputs; Jacobians are for vector outputs.
- Backpropagation efficiently computes vector-Jacobian products without materializing every Jacobian.
- Shape tracking is part of understanding matrix calculus in code.
- Coordinate-based models often need gradients with respect to both features and positions.

## Practical Checks

- What are the input and output shapes?
- Is the derivative a scalar, vector, matrix, Jacobian, or vector-Jacobian product?
- Are tensors treated as row vectors or column vectors in the notation?
- Does the implementation detach or stop gradients through part of the computation?

## Related

- [[concepts/math/calculus|Calculus]]
- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/geometric-deep-learning/coordinate-update|Coordinate update]]
