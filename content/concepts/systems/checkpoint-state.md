---
title: Checkpoint State
tags:
  - systems
  - checkpointing
  - reproducibility
---

# Checkpoint State

Checkpoint state is the full set of information needed to resume a training or inference workflow without changing its meaning. It is broader than model weights.

A training run at step $t$ can be viewed as state:

$$
S_t
=
\{\theta_t, o_t, s_t, r_t, c, d, e\}
$$

where $\theta_t$ are model parameters, $o_t$ is optimizer state, $s_t$ is scheduler state, $r_t$ is random state, $c$ is configuration, $d$ is dataset/preprocessing version, and $e$ is environment metadata.

Resume should approximate:

$$
\operatorname{Train}(S_t, k)
\approx
\operatorname{Train}(\operatorname{Restore}(\operatorname{Save}(S_t)), k)
$$

for the next $k$ steps, subject to nondeterministic hardware behavior.

## Required State

- Model weights and architecture config.
- Optimizer state.
- Scheduler state.
- Global step, epoch, consumed samples, and gradient accumulation position.
- Random number generator state.
- Mixed precision scaler state when applicable.
- Dataset version, split version, preprocessing version, and sampler state.
- Code commit, config, and environment metadata.

## Failure Modes

- Saving only weights and losing optimizer momentum.
- Resuming with a different dataset order.
- Selecting best checkpoints from test data.
- Corrupting a checkpoint on interrupted writes.
- Changing code or preprocessing between save and resume without recording it.

## Checks

- Can a run resume from the latest checkpoint and produce valid logs?
- Is checkpoint writing atomic?
- Is the best checkpoint chosen by validation metrics only?
- Is disk retention bounded?
- Are private paths and internal run names excluded from public notes?

## Related

- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/distributed-training|Distributed training]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[concepts/systems/failure-recovery|Failure recovery]]
