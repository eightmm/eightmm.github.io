---
title: Training Stability
tags:
  - machine-learning
  - optimization
  - training
---

# Training Stability

Training stability is whether optimization makes consistent progress without divergence, exploding gradients, collapsed representations, invalid outputs, or silent numerical failure. It connects [[concepts/machine-learning/loss-function|Loss function]], [[concepts/machine-learning/optimizer|Optimizer]], [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]], [[concepts/machine-learning/batch-size|Batch size]], and [[concepts/systems/training-run|Training run]].

A simple stability signal is the gradient norm:

$$
\lVert g_t\rVert_2
=
\left\lVert
\nabla_\theta \mathcal{L}_t(\theta_t)
\right\rVert_2
$$

where $g_t$ is the gradient at step $t$. Very large, very small, or highly erratic gradient norms often indicate an unstable setup.

## Common Causes

- Learning rate too high for the batch size or optimizer.
- Missing warmup for large models, large batches, or mixed precision.
- Loss scale mismatch across objectives or modalities.
- Bad initialization, normalization, or residual path behavior.
- Exploding gradients in sequence, recurrent, RL, or very deep models.
- Data preprocessing mismatch between train and validation.
- Incorrect model mode for dropout or normalization.
- Checkpoint resume without optimizer, scheduler, scaler, or random state.

## Stabilizers

- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]] with warmup and controlled decay.
- [[concepts/machine-learning/gradient-clipping|Gradient clipping]] when gradient spikes are expected.
- [[concepts/machine-learning/weight-decay|Weight decay]] and other regularization when overfitting or parameter growth matters.
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]] when memory limits require smaller micro-batches.
- Logging loss, gradient norm, learning rate, batch size, skipped steps, and validation metrics.
- Saving full [[concepts/systems/checkpoint-state|Checkpoint state]] for faithful resume.

## Instability vs Generalization

Stable training is not the same as good generalization:

$$
\text{good run}
\neq
\text{low train loss only}
$$

A run can be stable but overfit, or unstable but still occasionally produce a good validation checkpoint. Stability checks should be paired with validation metrics, leakage checks, and error analysis.

## Checks

- Are loss, learning rate, gradient norm, and validation metric logged together?
- Does instability start at warmup, after schedule changes, after resume, or after data changes?
- Is gradient clipping active often enough to change optimization dynamics?
- Is effective batch size computed correctly across devices and accumulation?
- Are optimizer state, scheduler state, scaler state, and random state restored on resume?
- Are train, validation, and inference preprocessing contracts identical where required?
- Is the failure a numerical issue, data issue, objective issue, architecture issue, or evaluation issue?

## Related

- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/gradient-clipping|Gradient clipping]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/observability|Observability]]
