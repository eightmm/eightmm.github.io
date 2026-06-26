---
title: Second-Order Optimization
tags:
  - machine-learning
  - optimization
  - curvature
---

# Second-Order Optimization

Second-order optimization uses curvature information, not only gradients, to choose parameter updates. In machine learning, this usually means using the Hessian or an approximation to it.

For a scalar loss $\mathcal{L}(\theta)$:

$$
g_t
=
\nabla_\theta \mathcal{L}(\theta_t),
\qquad
H_t
=
\nabla^2_\theta \mathcal{L}(\theta_t)
$$

Newton's method uses:

$$
\theta_{t+1}
=
\theta_t
-
H_t^{-1} g_t
$$

This rescales the gradient by local curvature.

## Local Quadratic View

Near $\theta_t$, a second-order approximation is:

$$
\mathcal{L}(\theta_t+\delta)
\approx
\mathcal{L}(\theta_t)
+
g_t^\top \delta
+
\frac{1}{2}\delta^\top H_t \delta
$$

Minimizing this quadratic gives the Newton step when $H_t$ is invertible and well-conditioned.

## Why Deep Learning Rarely Uses Full Hessians

| Issue | Meaning |
| --- | --- |
| Size | $H_t$ is $d\times d$ for $d$ parameters |
| Memory | full Hessian storage is usually impossible for large models |
| Cost | exact inversion is expensive |
| Indefiniteness | nonconvex losses can have negative curvature |
| Noise | mini-batch Hessians can be unstable |

Practical methods often use Hessian-vector products, diagonal approximations, low-rank approximations, quasi-Newton methods, natural-gradient approximations, or adaptive first-order optimizers.

## Curvature Diagnostics

Curvature still matters even when training uses AdamW or SGD:

| Signal | Interpretation |
| --- | --- |
| Large Hessian eigenvalues | sharp directions, learning-rate sensitivity |
| Ill-conditioning | some directions need much smaller steps than others |
| Negative curvature | saddle or nonconvex region |
| Large gradient norm with unstable loss | step may be too large for local curvature |
| Flat validation response | hyperparameter may not affect the selected metric strongly |

## Boundaries

Second-order information can support different claims:

| Use | Claim |
| --- | --- |
| Optimization method | curvature directly changes updates |
| Diagnostic | curvature explains instability or sharpness |
| Uncertainty approximation | local curvature approximates posterior shape |
| Architecture analysis | curvature compares trainability or sensitivity |

Do not treat a curvature plot as proof of generalization. It needs a metric, split, and uncertainty boundary.

## Checks

- Is the method using a full Hessian, Hessian-vector products, diagonal approximation, or a proxy?
- Is curvature computed on train, validation, or test data?
- Is the Hessian of loss, log likelihood, energy, or another objective?
- Are reported eigenvalues comparable across model scales and losses?
- Does the second-order claim affect optimization, uncertainty, or only interpretation?

## Related

- [[concepts/math/jacobian-hessian|Jacobian and Hessian]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/loss-landscape|Loss landscape]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/math/bayesian-inference|Bayesian inference]]
