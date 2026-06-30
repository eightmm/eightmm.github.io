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

## Coupled vs Decoupled

The same coefficient $\lambda$ can mean different things depending on the optimizer.

| Setup | Formula Shape | Interpretation |
| --- | --- | --- |
| $L_2$ objective penalty | $\hat{R}(\theta)+\frac{\lambda}{2}\|\theta\|_2^2$ | changes the optimized objective |
| SGD with $L_2$ | $\theta_{t+1}=(1-\eta\lambda)\theta_t-\eta g_t$ | equivalent to multiplicative shrinkage |
| Adam with $L_2$ | Adam receives $g_t+\lambda\theta_t$ | decay is scaled by adaptive denominator |
| AdamW | $\theta_{t+1}=(1-\eta\lambda)\theta_t-\eta u_t$ | direct decoupled shrinkage |

For plain SGD, coupled $L_2$ and weight decay are closely related. For adaptive optimizers, they should be treated as different training recipes.

## Effective Decay Over Training

With decoupled decay, a parameter is repeatedly multiplied by:

$$
1-\eta_t\lambda
$$

Ignoring gradient updates, after $T$ optimizer steps:

$$
\theta_T
\approx
\left[
\prod_{t=1}^{T}(1-\eta_t\lambda)
\right]\theta_0
$$

So effective decay depends on learning-rate schedule and optimizer step count, not only on $\lambda$.

## Parameter Group Contract

Weight decay should be documented as a parameter-group rule:

| Group | Typical Choice | Check |
| --- | --- | --- |
| matrix weights | decay | linear, convolution, attention projections |
| bias terms | no decay | otherwise shifts may be over-regularized |
| normalization parameters | no decay | scale/shift parameters control activation statistics |
| embeddings | depends | can dominate parameter count; tune separately |
| task heads | depends | small heads may need different decay |

The public note should state the rule, not private run identifiers or internal file paths.

## Interaction With Generalization

Weight decay can reduce overfitting, but it is not proof of better generalization:

$$
\lambda^\star
=
\arg\min_{\lambda}
M_{\mathrm{val}}(\theta_\lambda)
$$

The selected value depends on validation metric, split, schedule, batch size, optimizer, augmentation, and training length. If those change, the best $\lambda$ may change too.

## Checks

- Is weight decay coupled to the loss or decoupled in the optimizer?
- Are bias, normalization, and embedding parameters excluded when appropriate?
- Is the decay coefficient reported separately from learning rate?
- Does stronger decay improve validation performance or only reduce training loss?
- Is weight decay being confused with dropout or early stopping?
- Is the effective decay affected by schedule length or number of optimizer steps?
- Are parameter groups logged in the run record?

## Related

- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/adam|Adam]]
- [[concepts/machine-learning/adamw|AdamW]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/empirical-risk-minimization|Empirical risk minimization]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
