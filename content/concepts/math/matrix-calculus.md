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

See [[concepts/math/jacobian-hessian|Jacobian and Hessian]] for the relationship between Jacobians, Hessians, curvature, and automatic differentiation products.

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

## Shape Contract

Always record the derivative target, source, and shape:

| Expression | Shape | Meaning |
| --- | --- | --- |
| $x\in\mathbb{R}^{d}$ | $d$ | input vector |
| $y=f(x)\in\mathbb{R}^{m}$ | $m$ | vector output |
| $J_f(x)=\partial y/\partial x$ | $m\times d$ | local linear map from input changes to output changes |
| $\mathcal{L}(y)$ | scalar | training loss or objective |
| $\nabla_x\mathcal{L}$ | $d$ | direction of input sensitivity |
| $\nabla_\theta\mathcal{L}$ | same as $\theta$ | parameter update signal |

For a composition:

$$
z = g(y), \qquad y=f(x)
$$

the chain rule is:

$$
\frac{\partial z}{\partial x}
=
\frac{\partial z}{\partial y}
\frac{\partial y}{\partial x}
$$

The order of multiplication depends on the row/column convention, so shape checking is the safest way to read equations.

## Common Layer Derivatives

For a batch linear layer:

$$
Y = XW^\top + \mathbf{1}b^\top
$$

with $X\in\mathbb{R}^{B\times d_{\mathrm{in}}}$ and $W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$:

$$
\frac{\partial \mathcal{L}}{\partial W}
=
\left(\frac{\partial \mathcal{L}}{\partial Y}\right)^\top X
$$

and:

$$
\frac{\partial \mathcal{L}}{\partial X}
=
\frac{\partial \mathcal{L}}{\partial Y}W
$$

These equations are useful when debugging tensor shapes, custom layers, or paper notation.

## Autograd Products

Most frameworks compute products, not full derivative matrices:

| Product | Notation | Common Use |
| --- | --- | --- |
| VJP | $v^\top J$ | reverse-mode backpropagation |
| JVP | $Jv$ | forward sensitivity, neural ODEs, implicit methods |
| HVP | $Hv$ | curvature diagnostics and second-order methods |

If a paper says it uses a Jacobian or Hessian, check whether it materializes the matrix or only uses a product estimator.

## Coordinate Models

For coordinate outputs $X\in\mathbb{R}^{N\times 3}$, derivatives can be taken with respect to:

| Source | Example |
| --- | --- |
| parameters | $\nabla_\theta \mathcal{L}$ for training |
| coordinates | $\nabla_X E(X)$ for forces or refinement |
| time | $dX_t/dt$ for ODE or flow models |
| noise | sensitivity of generated coordinates to latent variables |

The derivative should respect the same symmetry contract as the model output.

## Constrained Variables

Some variables do not live in unconstrained Euclidean space. Examples include probability simplexes, unit vectors, rotation matrices, valid molecular graphs, and restrained coordinate sets.

For a feasible set $\mathcal{C}$:

$$
x\in\mathcal{C}
$$

an update may require projection:

$$
x_{t+1}
=
\Pi_{\mathcal{C}}
\left(
x_t-\eta\nabla_x f(x_t)
\right)
$$

or a reparameterization:

$$
x = \phi(z),
\qquad
\nabla_z f(\phi(z))
=
J_\phi(z)^\top \nabla_x f(x)
$$

This distinction is important when a paper claims valid molecules, normalized probabilities, rotations, or geometry constraints.

## Practical Checks

- What are the input and output shapes?
- Is the derivative a scalar, vector, matrix, Jacobian, or vector-Jacobian product?
- Are tensors treated as row vectors or column vectors in the notation?
- Does the implementation detach or stop gradients through part of the computation?
- Is the derivative with respect to inputs, parameters, coordinates, or time?
- Is the variable unconstrained, projected, or reparameterized?
- Does the paper need an exact derivative matrix, or only an autograd product?

## Related

- [[concepts/math/calculus|Calculus]]
- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/math/jacobian-hessian|Jacobian and Hessian]]
- [[concepts/math/constrained-optimization|Constrained optimization]]
- [[concepts/math/tensor-shape-notation|Tensor shape notation]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/geometric-deep-learning/coordinate-update|Coordinate update]]
