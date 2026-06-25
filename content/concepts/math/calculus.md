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

## Related

- [[concepts/math/matrix-calculus|Matrix calculus]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/generative-models/flow-matching|Flow matching]]
