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

The local geometry of the loss is controlled by [[concepts/math/jacobian-hessian|Jacobian and Hessian]] information: gradients give first-order direction, while Hessian curvature affects stability and step size.

With mini-batches, the gradient is estimated from a batch $B$:

$$
g_t
=
\frac{1}{|B|}
\sum_{i\in B}
\nabla_\theta
\mathcal{L}(f_\theta(x_i),y_i)
$$

## Checks

- Is the gradient computed from the intended loss and valid examples only?
- Is the learning rate schedule stable?
- Are gradients clipped, accumulated, synchronized, or scaled?
- Does lower training loss improve validation performance?

## Related

- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/math/jacobian-hessian|Jacobian and Hessian]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/training-loop|Training loop]]
