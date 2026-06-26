---
title: Gradient Checking
tags:
  - machine-learning
  - optimization
  - verification
---

# Gradient Checking

Gradient checking compares an automatically computed gradient with a numerical finite-difference estimate. It is a debugging tool for custom losses, custom layers, coordinate transforms, masking logic, and scientific model code.

For a scalar objective $L(\theta)$, the central finite-difference estimate for parameter $\theta_j$ is:

$$
\frac{\partial L}{\partial \theta_j}
\approx
\frac{
L(\theta + \epsilon e_j)
-
L(\theta - \epsilon e_j)
}{
2\epsilon
}
$$

where $e_j$ is a unit vector and $\epsilon$ is a small perturbation.

## Relative Error

A common comparison is:

$$
\mathrm{relerr}
=
\frac{
\lVert g_{\mathrm{auto}} - g_{\mathrm{num}} \rVert
}{
\max(
\lVert g_{\mathrm{auto}} \rVert,
\lVert g_{\mathrm{num}} \rVert,
\epsilon_{\mathrm{safe}}
)
}
$$

The threshold depends on dtype, function smoothness, reduction scale, and numerical conditioning. The point is not to prove all training is correct; it is to catch broken derivative paths.

## When to Use

- A custom differentiable operation is added.
- A loss has masks, weights, or reductions.
- A coordinate transform affects geometry-sensitive outputs.
- A constraint, projection, or reparameterization affects the optimized variable.
- A model uses manual gradient manipulation.
- Training diverges before basic gradients have been validated.

## Pitfalls

- Finite differences are unreliable around discontinuities, argmax, hard thresholds, indexing, and random branches.
- Float32 can be too noisy; use double precision when possible.
- Stochastic layers should be disabled or seeded.
- Regularization terms and loss reductions must match exactly.
- Checking every parameter is expensive; sample representative elements.
- Projected or constrained updates may be nondifferentiable at boundaries.

## Constrained Variables

For a reparameterized constrained variable:

$$
x=\phi(z)
$$

the gradient check should usually compare derivatives with respect to $z$, not only $x$:

$$
\frac{\partial L(\phi(z))}{\partial z}
$$

For projected updates:

$$
\Pi_{\mathcal{C}}(x)
$$

finite differences can fail near the boundary of $\mathcal{C}$. In that case, check interior points, smooth penalty versions, or task-level invariants.

## Checks

- Is the model in deterministic mode?
- Is the same scalar loss used for both automatic and numerical gradients?
- Is the perturbation $\epsilon$ appropriate for the dtype and scale?
- Are masks and constraints applied identically in both evaluations?
- Are gradients checked before running a long experiment?
- If variables are constrained, is the check performed in the right parameterization?

## Related

- [[concepts/machine-learning/automatic-differentiation|Automatic differentiation]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/math/constrained-optimization|Constrained optimization]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/systems/experiment-lifecycle|Experiment lifecycle]]
- [[concepts/research-methodology/minimum-viable-experiment|Minimum viable experiment]]
