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

An informal variance relationship is:

$$
\operatorname{Var}(g_t)
\approx
\frac{1}{|B_t|}
\Sigma_g
$$

where $\Sigma_g$ is the per-example gradient covariance. Larger batches usually reduce gradient noise, but they also change memory use, throughput, optimizer dynamics, and sometimes generalization.

## Biased Estimates

The stochastic gradient is unbiased only under the sampling distribution assumed by the objective. If the sampler uses weights $q_i$ while the intended empirical risk is uniform, the expected gradient becomes:

$$
\mathbb{E}_{i\sim q}
\left[
\nabla_\theta \mathcal{L}_i(\theta)
\right]
=
\sum_{i=1}^{n}
q_i
\nabla_\theta \mathcal{L}_i(\theta)
$$

This is a different objective unless importance weights correct it:

$$
g
=
\frac{1}{|B|}
\sum_{i\in B}
\frac{1/n}{q_i}
\nabla_\theta \mathcal{L}_i(\theta)
$$

Class-balanced sampling, hard-example mining, padding masks, token weighting, and multi-task sampling all change what the gradient estimates.

## Distributed Averaging

With $N$ workers, each worker $k$ computes a local mini-batch gradient $g_t^{(k)}$. Synchronous data parallel training usually averages:

$$
g_t
=
\frac{1}{N}
\sum_{k=1}^{N}
g_t^{(k)}
$$

This is equivalent to a larger global batch only if workers use compatible data shards, loss normalization, mixed-precision scaling, and accumulation boundaries.

## Checks

- Is the mini-batch sampled from the intended distribution?
- Are class weights, masks, or sampling weights changing the gradient estimate?
- Is gradient accumulation equivalent to the intended batch size?
- Are distributed workers averaging gradients correctly?
- Is gradient noise helping generalization or causing instability?
- Is the loss normalized by examples, tokens, positive labels, or valid elements?
- Are rare classes oversampled for training but evaluated under natural prevalence?
- Are compared runs using the same effective batch and learning-rate schedule?

## Related

- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/systems/distributed-training|Distributed training]]
- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/math/expectation|Expectation]]
