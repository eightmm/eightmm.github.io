---
title: Stochastic Gradient
tags:
  - machine-learning
  - optimization
---

# Stochastic Gradient

A stochastic gradient estimates the full training gradient using one example or a mini-batch. It is the reason large models can be trained without evaluating the entire dataset at every step.

For empirical risk:

$$
\hat{R}(\theta)
= \frac{1}{n}
\sum_{i=1}^{n}
\mathcal{L}_i(\theta)
$$

the full gradient is:

$$
\nabla_\theta \hat{R}(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\nabla_\theta \mathcal{L}_i(\theta)
$$

For a mini-batch $B_t$:

$$
g_t
=
\frac{1}{|B_t|}
\sum_{i \in B_t}
\nabla_\theta \mathcal{L}_i(\theta_t)
$$

$g_t$ is a stochastic estimate of the full gradient at step $t$.

## Noise and Batch Size

If mini-batches are sampled uniformly:

$$
\mathbb{E}[g_t]
=
\nabla_\theta \hat{R}(\theta_t)
$$

but the variance depends on batch size and data heterogeneity. This connects directly to [[concepts/machine-learning/batch-size|batch size]], learning rate, and training stability.

## Checks

- Is the mini-batch sampled from the intended distribution?
- Are class weights, masks, or sampling weights changing the gradient estimate?
- Is gradient accumulation equivalent to the intended batch size?
- Are distributed workers averaging gradients correctly?
- Is gradient noise helping generalization or causing instability?

## Related

- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/math/expectation|Expectation]]
