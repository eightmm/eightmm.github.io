---
title: Training Loop
tags:
  - machine-learning
  - optimization
---

# Training Loop

A training loop repeatedly samples data, computes predictions, evaluates a loss, backpropagates gradients, and updates parameters.

One iteration can be summarized as:

$$
\hat{y} = f_\theta(x)
$$

$$
\mathcal{L} = \mathcal{L}(\hat{y}, y)
$$

$$
g = \nabla_\theta \mathcal{L}
$$

$$
\theta \leftarrow \operatorname{Optimizer}(\theta, g)
$$

## Core Steps

- Load a batch from the training split.
- Run the forward pass.
- Compute the loss.
- Backpropagate gradients.
- Update parameters.
- Periodically evaluate on validation data.

## Checks

- Are train, validation, and test splits separated before preprocessing?
- Are gradients zeroed or accumulated intentionally?
- Are model modes set correctly for dropout and normalization?
- Are checkpoint, seed, and metric logging sufficient to reproduce the run?

## Related

- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/regularization|Regularization]]
