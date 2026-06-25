---
title: Weight Decay
tags:
  - machine-learning
  - optimization
  - regularization
---

# Weight Decay

Weight decay penalizes large parameters during optimization. It is commonly used as a form of regularization in neural network training.

Classical $L_2$ regularization adds a penalty to the objective:

$$
J(\theta)
=
\hat{R}(\theta)
+ \frac{\lambda}{2}
\lVert \theta \rVert_2^2
$$

The gradient becomes:

$$
\nabla_\theta J(\theta)
=
\nabla_\theta \hat{R}(\theta)
+ \lambda \theta
$$

For plain SGD, this gives:

$$
\theta_{t+1}
=
\theta_t
- \eta
\left(
g_t + \lambda \theta_t
\right)
$$

$\eta$ is the learning rate, $g_t$ is the data gradient, and $\lambda$ is the decay coefficient.

## Decoupled Weight Decay

Adaptive optimizers often use decoupled weight decay:

$$
\theta_{t+1}
=
(1 - \eta\lambda)\theta_t
- \eta u_t
$$

$u_t$ is the optimizer update direction. This separates parameter shrinkage from adaptive gradient normalization.

## Checks

- Is weight decay coupled to the loss or decoupled in the optimizer?
- Are bias, normalization, and embedding parameters excluded when appropriate?
- Is the decay coefficient reported separately from learning rate?
- Does stronger decay improve validation performance or only reduce training loss?
- Is weight decay being confused with dropout or early stopping?

## Related

- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/empirical-risk-minimization|Empirical risk minimization]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
