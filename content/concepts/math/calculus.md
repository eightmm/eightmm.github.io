---
title: Calculus
tags:
  - math
  - calculus
  - optimization
---

# Calculus

Calculus studies how quantities change. In machine learning, calculus appears whenever a loss changes with parameters, inputs, time, noise level, or coordinates.

For a scalar function $f:\mathbb{R}\to\mathbb{R}$, the derivative is:

$$
\frac{df}{dx}
=
\lim_{\epsilon\to 0}
\frac{f(x+\epsilon)-f(x)}{\epsilon}
$$

For a multivariate function $f:\mathbb{R}^d\to\mathbb{R}$, the gradient is:

$$
\nabla_x f
=
\left[
\frac{\partial f}{\partial x_1},
\ldots,
\frac{\partial f}{\partial x_d}
\right]^\top
$$

The chain rule is the core of backpropagation:

$$
\frac{d}{dx}f(g(x))
=
f'(g(x))g'(x)
$$

See [[concepts/math/chain-rule|Chain rule]] for composition, vector-Jacobian products, and backpropagation notation.

## Jacobian View

For a vector-valued function $f:\mathbb{R}^d\to\mathbb{R}^k$, the derivative is the Jacobian:

$$
J_f(x)
=
\frac{\partial f}{\partial x}
\in \mathbb{R}^{k\times d},
\qquad
(J_f)_{ij}
=
\frac{\partial f_i}{\partial x_j}
$$

For a composition $h(x)=f(g(x))$:

$$
J_h(x)
=
J_f(g(x))J_g(x)
$$

Backpropagation usually avoids materializing the full Jacobian. It propagates vector-Jacobian products:

$$
v^\top J_f(x)
$$

from the output loss back toward earlier variables.

## Directional Derivative

The directional derivative of $f$ at $x$ in direction $u$ is:

$$
D_u f(x)
=
\lim_{\epsilon\to 0}
\frac{f(x+\epsilon u)-f(x)}{\epsilon}
=
\nabla_x f(x)^\top u
$$

This explains why the gradient gives the steepest local increase under an $\ell_2$ direction constraint:

$$
\max_{\|u\|_2=1} \nabla_x f(x)^\top u
=
\|\nabla_x f(x)\|_2
$$

The maximizing direction is $u=\nabla_x f(x)/\|\nabla_x f(x)\|_2$ when the gradient is nonzero.

Second-order information describes curvature. For a scalar loss $\mathcal{L}$, this is captured by the Hessian:

$$
H_{\mathcal{L}}(x)
=
\nabla_x^2\mathcal{L}(x)
$$

See [[concepts/math/jacobian-hessian|Jacobian and Hessian]] for vector-valued derivatives, curvature, and Jacobian/Hessian products.

For a small perturbation $\Delta x$, a second-order approximation is:

$$
f(x+\Delta x)
\approx
f(x)
+ \nabla f(x)^\top \Delta x
+ \frac{1}{2}\Delta x^\top H_f(x)\Delta x
$$

This is the local picture behind curvature, Newton-style updates, sharpness analysis, and some stability arguments.

## Gradient Descent Connection

For parameters $\theta$ and loss $\mathcal{L}(\theta)$, the simplest update is:

$$
\theta_{t+1}
=
\theta_t
- \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

where $\eta$ is the learning rate. In stochastic training, the full gradient is replaced by an estimate from a mini-batch:

$$
g_t
=
\frac{1}{B}\sum_{i\in \mathcal{B}_t}
\nabla_\theta \mathcal{L}(f_\theta(x_i), y_i)
$$

The math note defines the derivative object. Optimizer behavior belongs in [[concepts/machine-learning/optimization|Optimization]] and [[concepts/machine-learning/optimizer|Optimizer]].

## Key Ideas

- Derivatives describe local sensitivity.
- Gradients point in the direction of steepest local increase for scalar functions.
- Optimization usually moves against the gradient of a loss.
- Chain rules compose derivatives across layers, time steps, or computational graphs.
- Continuous-time models, diffusion, flow matching, and dynamics use derivatives with respect to time or noise level.

## Practical Checks

- Is the function scalar-valued or vector-valued?
- Are derivatives taken with respect to inputs, parameters, time, or coordinates?
- Is the gradient used for optimization, sensitivity analysis, or dynamics?
- Does the implementation compute exact gradients, approximate finite differences, or stopped gradients?
- Is the code using gradients, Jacobian-vector products, vector-Jacobian products, or Hessian-vector products?
- Does the notation distinguish population loss, mini-batch loss, and single-example loss?

## Related

- [[concepts/math/matrix-calculus|Matrix calculus]]
- [[concepts/math/chain-rule|Chain rule]]
- [[concepts/math/jacobian-hessian|Jacobian and Hessian]]
- [[concepts/math/vector-norm-similarity|Vector norm and similarity]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/stochastic-gradient|Stochastic gradient]]
- [[concepts/generative-models/flow-matching|Flow matching]]
