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

## Run Contract

A run is reproducible only if the executable state and data state are tied together.

$$
\text{run}
=
(\text{code},\ \text{config},\ \text{data},\ \text{environment},\ \text{seed},\ \text{artifacts},\ \text{metrics})
$$

| Field | Record |
| --- | --- |
| code | commit, diff policy, entrypoint |
| config | model, optimizer, schedule, batch, precision |
| data | dataset version, split, preprocessing contract |
| environment | container/module, package versions, hardware class |
| seed | global seed, dataloader seed, nondeterminism policy |
| artifacts | checkpoints, logs, predictions, config snapshot |
| metrics | train/val/test separation and step axis |

## State Machine

Training runs are not just `started` or `done`.

| State | Evidence |
| --- | --- |
| planned | config and resource request exist |
| launched | job id/process id and initial log exist |
| running | metrics or checkpoints advance |
| interrupted | failure/preemption/timeout evidence exists |
| resumed | checkpoint state and config compatibility checked |
| completed | final artifact and validation summary exist |
| invalidated | leakage, bad config, corrupted data, or wrong environment found |

Invalidated runs should remain visible in private or public-safe logs if they prevent repeating a mistake.

## Run State

- Code commit and uncommitted diff policy.
- Dataset version and preprocessing version.
- Model config and training config.
- Seed and deterministic settings.
- Optimizer, scheduler, and mixed-precision settings.
- Checkpoint, metric log, prediction, and artifact policy.

## Step Accounting

Metrics should share a clear step axis:

$$
\text{global step}
=
\text{optimizer updates}
\neq
\text{batches seen}
\neq
\text{tokens or examples seen}
$$

This distinction matters with gradient accumulation, distributed training, variable batch sizes, or resume.

| Axis | Use |
| --- | --- |
| optimizer step | learning rate schedule and checkpointing |
| sample count | data exposure and comparison across batch sizes |
| token/residue/atom count | variable-length domains |
| wall time | systems throughput and cost |
| epoch | only meaningful if dataset pass is well-defined |

## Checks

- Can the run resume after interruption?
- Are public [[concepts/systems/run-artifact|run artifacts]] sufficient for later inspection?
- Are [[concepts/machine-learning/training-stability|training stability]] signals logged with the run?
- Are [[concepts/machine-learning/learning-curve|learning curves]] and validation metrics recorded on the same step axis?
- Are validation metrics separated from training loss?
- Is the best checkpoint selected only from validation data?
- Are failed runs recorded with enough context to learn from them?
- Are public notes free of private paths, unpublished results, and internal task names?
- Is the run state completed, failed, interrupted, resumed, or invalidated?
- Are metrics comparable on the same step/sample/time axis?

## Related

- [[concepts/systems/experiment-lifecycle|Experiment lifecycle]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/learning-curve|Learning curve]]
- [[concepts/machine-learning/validation-curve|Validation curve]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/systems/distributed-training|Distributed training]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[infra/hpc/checkpointing|Checkpointing]]
