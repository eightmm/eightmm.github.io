---
title: Optimization
tags:
  - machine-learning
---

# Optimization

Optimization is the process of updating model parameters to improve an objective. In machine learning, the optimized objective is usually a proxy for the real task.

Gradient descent updates parameters in the direction that locally reduces the objective:

$$
\theta_{t+1}
= \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

Here $\eta$ is the learning rate.

The objective can be viewed as a [[concepts/machine-learning/loss-landscape|Loss landscape]]:

$$
\theta \mapsto \mathcal{L}(\theta)
$$

Optimization follows local information from gradients, while evaluation decides whether that local progress supports the intended generalization claim.

## Core Ideas

- Loss defines the training signal.
- Gradients indicate local directions for parameter updates.
- Learning rate controls update size.
- Batches estimate the objective from subsets of data.
- Optimizer state can change the update beyond raw gradients.
- Curvature and scale affect how stable a step size is.
- Second-order methods or diagnostics use curvature information directly or approximately.

## Update View

A generic first-order optimizer can be written as:

$$
g_t
=
\nabla_\theta \mathcal{L}(\theta_t)
$$

$$
s_t
=
\operatorname{StateUpdate}(s_{t-1}, g_t)
$$

$$
\theta_{t+1}
=
\operatorname{Update}(\theta_t, g_t, s_t, \eta_t)
$$

For plain gradient descent, $s_t$ is empty. For [[concepts/machine-learning/adam|Adam]] and [[concepts/machine-learning/adamw|AdamW]], $s_t$ contains moment estimates.

## Curvature View

First-order methods use gradients. Second-order methods also use curvature:

$$
H_t
=
\nabla^2_\theta \mathcal{L}(\theta_t)
$$

The Hessian is usually too large to materialize for deep models, but Hessian-vector products, diagonal approximations, and curvature diagnostics still help explain unstable training and learning-rate sensitivity.

## Watch For

- Lower training loss does not guarantee better generalization.
- Optimization instability can look like model failure.
- Hyperparameters can dominate small experiments.
- Broken gradients can look like a bad optimizer.
- A scheduler stepped at the wrong boundary can silently change the experiment.

## Related

- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/machine-learning/automatic-differentiation|Automatic differentiation]]
- [[concepts/machine-learning/gradient-checking|Gradient checking]]
- [[concepts/machine-learning/loss-landscape|Loss landscape]]
- [[concepts/machine-learning/second-order-optimization|Second-order optimization]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/evaluation/index|Evaluation]]
