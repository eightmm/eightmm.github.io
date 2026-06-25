---
title: Training Run
tags:
  - systems
  - training
  - reproducibility
---

# Training Run

A training run is one execution of a training procedure with a fixed code version, configuration, dataset version, seed policy, and environment.

Empirical risk minimization for one run is:

$$
\hat{\theta}
= \arg\min_\theta
\frac{1}{n}\sum_{i=1}^{n}\mathcal{L}(f_\theta(x_i),y_i)
$$

but the actual run also includes optimizer state, scheduler state, random seeds, checkpoints, logging, and hardware behavior.

## Run State

- Code commit and uncommitted diff policy.
- Dataset version and preprocessing version.
- Model config and training config.
- Seed and deterministic settings.
- Optimizer, scheduler, and mixed-precision settings.
- Checkpoint, metric log, prediction, and artifact policy.

## Checks

- Can the run resume after interruption?
- Are public [[concepts/systems/run-artifact|run artifacts]] sufficient for later inspection?
- Are [[concepts/machine-learning/training-stability|training stability]] signals logged with the run?
- Are [[concepts/machine-learning/learning-curve|learning curves]] and validation metrics recorded on the same step axis?
- Are validation metrics separated from training loss?
- Is the best checkpoint selected only from validation data?
- Are failed runs recorded with enough context to learn from them?
- Are public notes free of private paths, unpublished results, and internal task names?

## Related

- [[concepts/systems/experiment-lifecycle|Experiment lifecycle]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/learning-curve|Learning curve]]
- [[concepts/machine-learning/validation-curve|Validation curve]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/systems/distributed-training|Distributed training]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[infra/hpc/checkpointing|Checkpointing]]
