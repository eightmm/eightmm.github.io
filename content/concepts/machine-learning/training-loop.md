---
title: Training Loop
tags:
  - machine-learning
  - optimization
---

# Training Loop

A training loop repeatedly samples data, computes predictions, evaluates a loss, backpropagates gradients, and updates parameters.

The loop optimizes empirical risk over mini-batches:

$$
\hat{R}_B(\theta)
=
\frac{1}{|B|}
\sum_{i\in B}
\ell(f_\theta(x_i), y_i)
$$

One micro-step can be summarized as:

$$
\hat{y} = f_\theta(x)
$$

$$
\mathcal{L} = \mathcal{L}(\hat{y}, y)
$$

$$
g = \nabla_\theta \mathcal{L}
$$

One optimizer step updates parameters:

$$
\theta \leftarrow \operatorname{Optimizer}(\theta, g)
$$

When gradients are accumulated, several micro-batches contribute to $g$ before the update. This changes how [[concepts/machine-learning/training-step-accounting|training step accounting]], logging, schedules, and checkpoints should be counted.

## Core Steps

- Load a batch from the training split.
- Run the forward pass.
- Compute the loss.
- Backpropagate gradients through [[concepts/machine-learning/automatic-differentiation|Automatic differentiation]].
- Accumulate or reduce gradients across micro-batches and devices.
- Optionally unscale, clip, or inspect gradients.
- Update parameters with the optimizer.
- Step the scheduler at the intended boundary.
- Zero gradients at the intended boundary.
- Periodically evaluate on validation data.

## Canonical Deep Learning Loop

```text
for batch in train_loader:
    model.train()
    output = model(batch.x)
    loss = loss_fn(output, batch.y) / accumulation_steps
    loss.backward()

    if ready_for_optimizer_step:
        clip_gradients_if_needed()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        log_train_state()

    if ready_for_validation:
        model.eval()
        run_validation_without_gradients()
```

The key point is that loss computation, backward pass, optimizer update, scheduler update, validation, and checkpointing do not always share the same step boundary.

## State Boundary

At optimizer step $t$, a training run can be viewed as:

$$
S_t
=
\{\theta_t, o_t, \eta_t, r_t, d_t, c\}
$$

where $\theta_t$ is model state, $o_t$ is optimizer state, $\eta_t$ is scheduler state, $r_t$ is random state, $d_t$ is data-loader or sampler state, and $c$ is configuration. A faithful resume should restore this state, not only model weights.

## Checks

- Are train, validation, and test splits separated before preprocessing?
- Are gradients zeroed or accumulated intentionally?
- Is [[concepts/machine-learning/gradient-accumulation|gradient accumulation]] counted in optimizer steps, logging, and schedules?
- Is [[concepts/machine-learning/training-step-accounting|training step accounting]] explicit in config and logs?
- Are model modes set correctly for dropout and normalization?
- Are stability signals such as loss, learning rate, and gradient norm logged?
- Does [[concepts/machine-learning/gradient-checking|Gradient checking]] pass before long runs when custom differentiable code is used?
- Are optimizer state, scheduler state, and mixed-precision state saved when resuming matters?
- Are checkpoint, seed, and metric logging sufficient to reproduce the run?

## Related

- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/machine-learning/automatic-differentiation|Automatic differentiation]]
- [[concepts/machine-learning/gradient-checking|Gradient checking]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/adam|Adam]]
- [[concepts/machine-learning/adamw|AdamW]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/regularization|Regularization]]
