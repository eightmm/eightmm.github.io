---
title: Loss Landscape
tags:
  - machine-learning
  - optimization
---

# Loss Landscape

A loss landscape is the objective value as a function of model parameters. It is a useful mental model for optimization, even though modern neural networks live in very high-dimensional spaces.

For empirical risk:

$$
\hat{R}(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\ell(f_\theta(x_i), y_i)
$$

the landscape is the surface $\theta \mapsto \hat{R}(\theta)$.

## Local Geometry

Near a point $\theta_t$, a second-order approximation is:

$$
\hat{R}(\theta_t + \Delta)
\approx
\hat{R}(\theta_t)
+
\nabla \hat{R}(\theta_t)^\top \Delta
+
\frac{1}{2}
\Delta^\top
H(\theta_t)
\Delta
$$

where $H(\theta_t)$ is the Hessian. The gradient gives local slope; the Hessian describes local curvature.

This connects directly to [[concepts/machine-learning/gradient-descent|Gradient descent]]:

$$
\theta_{t+1}
=
\theta_t
-
\eta_t
\nabla \hat{R}(\theta_t)
$$

If $\eta_t$ is too large relative to curvature, the update can overshoot or diverge.

## Useful Distinctions

- Local minimum: nearby perturbations increase loss.
- Saddle point: some directions increase loss and others decrease loss.
- Plateau: gradients are small over a region.
- Sharp region: loss changes quickly under small parameter perturbations.
- Flat region: loss changes slowly under small parameter perturbations.

These labels are diagnostic tools, not direct guarantees of generalization. Reparameterization, normalization, scale symmetries, and optimizer state can change how the same model appears in parameter space.

## Diagnostics

- Training loss decreases smoothly: optimization is making local progress.
- Loss decreases but validation worsens: optimization may be overfitting the train distribution.
- Loss spikes: learning rate, gradient scaling, batch composition, or data corruption may be unstable.
- Gradient norm collapses: saturation, masking, detach, or bad initialization may block learning.
- Gradient norm explodes: curvature, recurrent depth, bad normalization, or outlier batches may dominate.

## Checks

- Is instability caused by optimization, data, objective, or implementation?
- Are loss, gradient norm, learning rate, and batch statistics logged together?
- Does a smaller learning rate or simpler model remove the failure?
- Does [[concepts/machine-learning/gradient-checking|Gradient checking]] rule out broken derivatives for custom code?
- Is the claim based on validation behavior, not only training loss?

## Related

- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/math/jacobian-hessian|Jacobian and Hessian]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/learning-curve|Learning curve]]
- [[concepts/machine-learning/generalization|Generalization]]
