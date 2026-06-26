---
title: Gradient Descent
tags:
  - machine-learning
  - optimization
---

# Gradient Descent

Gradient descent updates parameters in the direction that locally decreases a loss. It is the basic optimization idea behind most neural network training.

The update is:

$$
\theta_{t+1}
=
\theta_t
- \eta_t
\nabla_\theta \mathcal{L}(\theta_t)
$$

where $\theta_t$ are parameters, $\eta_t$ is the learning rate, and $\nabla_\theta\mathcal{L}$ is the gradient.

More generally, training updates an objective $J(\theta)$:

$$
J(\theta)
=
\hat{R}(\theta)
+
\lambda \Omega(\theta)
$$

and the update is:

$$
\theta_{t+1}
=
\theta_t
-
\eta_t
\nabla_\theta J(\theta_t)
$$

where $\hat{R}$ is empirical risk, $\Omega$ is a regularizer, and $\lambda$ controls regularization strength.

The local geometry of the loss is controlled by [[concepts/math/jacobian-hessian|Jacobian and Hessian]] information: gradients give first-order direction, while Hessian curvature affects stability and step size.

This geometry is often described as the [[concepts/machine-learning/loss-landscape|Loss landscape]]. Gradient descent only uses first-order local information, so the same update rule can behave differently in flat, sharp, noisy, or badly scaled regions.

## Direction and Step Size

For a small perturbation $\Delta\theta$, first-order approximation gives:

$$
J(\theta+\Delta\theta)
\approx
J(\theta)
+
\nabla_\theta J(\theta)^\top \Delta\theta
$$

Choosing $\Delta\theta=-\eta\nabla_\theta J(\theta)$ gives:

$$
J(\theta+\Delta\theta)
\approx
J(\theta)
-
\eta
\left\|
\nabla_\theta J(\theta)
\right\|_2^2
$$

This explains why the negative gradient is locally downhill. It does not guarantee global improvement when the step is too large, gradients are noisy, or the objective is non-convex.

With mini-batches, the gradient is estimated from a batch $B$:

$$
g_t
=
\frac{1}{|B|}
\sum_{i\in B}
\nabla_\theta
\mathcal{L}(f_\theta(x_i),y_i)
$$

Then the practical update is:

$$
\theta_{t+1}
=
\theta_t
-
\eta_t
g_t
$$

possibly after gradient clipping, mixed-precision scaling, distributed averaging, or optimizer-specific transformations.

## Failure Modes

- Overshooting: $\eta_t$ is too large, so loss oscillates or diverges.
- Slow progress: $\eta_t$ is too small or curvature is badly scaled.
- Bad gradient estimate: masking, padding, sampling, or distributed averaging changes $g_t$.
- Train-only improvement: training loss decreases while validation loss or target metric does not.
- Resume drift: optimizer, scheduler, scaler, or random sampler state is not restored.

## Checks

- Is the gradient computed from the intended loss and valid examples only?
- Is the learning rate schedule stable?
- Are gradients clipped, accumulated, synchronized, or scaled?
- Does lower training loss improve validation performance?
- Is the x-axis counted in optimizer steps, samples, tokens, or epochs?
- Is the same update boundary used across compared runs?
- Are optimizer and scheduler states restored when resuming?

## Related

- [[concepts/machine-learning/empirical-risk-minimization|Empirical risk minimization]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/machine-learning/automatic-differentiation|Automatic differentiation]]
- [[concepts/machine-learning/loss-landscape|Loss landscape]]
- [[concepts/math/jacobian-hessian|Jacobian and Hessian]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/gradient-clipping|Gradient clipping]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
- [[concepts/machine-learning/training-loop|Training loop]]
