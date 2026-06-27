---
title: Chain Rule
tags:
  - math
  - calculus
  - gradients
---

# Chain Rule

The chain rule tells how derivatives pass through a composition of functions. It is the basic mathematical rule behind backpropagation.

For scalar functions:

$$
y=f(h),
\qquad
h=g(x)
$$

the derivative is:

$$
\frac{dy}{dx}
=
\frac{dy}{dh}
\frac{dh}{dx}
=
f'(g(x))g'(x)
$$

The important idea is that a change in $x$ first changes $h$, and the change in $h$ then changes $y$.

## Neural Network View

For a model layer:

$$
h_{\ell+1}=f_\ell(h_\ell;\theta_\ell)
$$

and a scalar loss $\mathcal{L}$, backpropagation repeatedly applies:

$$
\frac{\partial \mathcal{L}}{\partial h_\ell}
=
\frac{\partial \mathcal{L}}{\partial h_{\ell+1}}
\frac{\partial h_{\ell+1}}{\partial h_\ell}
$$

and:

$$
\frac{\partial \mathcal{L}}{\partial \theta_\ell}
=
\frac{\partial \mathcal{L}}{\partial h_{\ell+1}}
\frac{\partial h_{\ell+1}}{\partial \theta_\ell}
$$

This is why deep learning frameworks can compute gradients by traversing a computational graph backward from the loss.

## Vector Form

For vector-valued functions:

$$
z=g(y),
\qquad
y=f(x)
$$

the Jacobian chain rule is:

$$
\frac{\partial z}{\partial x}
=
\frac{\partial z}{\partial y}
\frac{\partial y}{\partial x}
$$

Shape checking is the safest way to read this equation. If $x\in\mathbb{R}^n$, $y\in\mathbb{R}^m$, and $z\in\mathbb{R}^k$, then:

$$
\frac{\partial z}{\partial y}\in\mathbb{R}^{k\times m},
\qquad
\frac{\partial y}{\partial x}\in\mathbb{R}^{m\times n},
\qquad
\frac{\partial z}{\partial x}\in\mathbb{R}^{k\times n}
$$

## Why It Matters

- Backpropagation is repeated chain rule.
- Normalizing flows need Jacobian terms from invertible transformations.
- Coordinate refinement and force prediction need gradients through geometry.
- Recurrent models and diffusion samplers apply chain-like updates across time steps.
- Detach or stop-gradient operations intentionally break part of the chain.

## Checks

- What is the outer function and what is the inner function?
- Is the output scalar or vector-valued?
- Are the derivative shapes compatible?
- Does the computation include a detach, discrete sample, mask, or non-differentiable operation?
- Is the chain rule being used for training, sensitivity analysis, or a change of variables?

## Related

- [[math/calculus-gradients|Calculus and gradients]]
- [[concepts/math/calculus|Calculus]]
- [[concepts/math/matrix-calculus|Matrix calculus]]
- [[concepts/math/jacobian-hessian|Jacobian and Hessian]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
