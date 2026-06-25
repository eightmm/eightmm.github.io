---
title: Empirical Risk Minimization
tags:
  - machine-learning
  - optimization
---

# Empirical Risk Minimization

Empirical risk minimization is the basic training principle of choosing parameters that minimize average loss on observed data.

Given a dataset:

$$
\mathcal{D}
= \{(x_i, y_i)\}_{i=1}^{n}
$$

the empirical risk is:

$$
\hat{R}(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathcal{L}(f_\theta(x_i), y_i)
$$

Training chooses:

$$
\hat{\theta}
=
\arg\min_{\theta}
\hat{R}(\theta)
$$

$x_i$ is an input, $y_i$ is a target, $f_\theta$ is the model, and $\mathcal{L}$ is the loss.

## Why It Matters

Many learning methods are variations on this template. The difference is usually in the data distribution, loss, regularizer, sampling process, or optimization method.

With a regularizer:

$$
\hat{\theta}
=
\arg\min_{\theta}
\left[
\frac{1}{n}
\sum_{i=1}^{n}
\mathcal{L}(f_\theta(x_i), y_i)
+ \lambda \Omega(\theta)
\right]
$$

$\Omega(\theta)$ penalizes model complexity or parameter size, and $\lambda$ controls the penalty strength.

## Checks

- What examples define the empirical distribution?
- Does the loss match the task and target semantics?
- Is the training objective the same quantity as the evaluation metric?
- Are weights, masks, or sampling probabilities changing the risk?
- Is regularization explicit or hidden in the optimizer?

## Related

- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/stochastic-gradient|Stochastic gradient]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
