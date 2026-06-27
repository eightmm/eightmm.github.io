---
title: Checkpointing
tags:
  - infra
  - hpc
  - training
---

# Checkpointing

Checkpointing periodically saves training state so a job can resume after preemption, failure, or a wall-clock limit. A checkpoint should restore the run exactly, not just the weights.

The checkpoint state can be written as:

$$
C_t
=
(\theta_t,\ o_t,\ s_t,\ k_t,\ r_t,\ e_t)
$$

where $\theta_t$ is model state, $o_t$ optimizer state, $s_t$ scheduler state, $k_t$ step or epoch, $r_t$ RNG state, and $e_t$ environment/run metadata.

## Resume Contract

A resume should satisfy:

$$
\operatorname{resume}(C_t)
\rightarrow
\text{same training state at step } t
$$

This means the next batch, learning-rate schedule, gradient-scaler state, distributed rank behavior, and random augmentations should be consistent with the intended run policy.

## Atomic Write

Checkpoint writes should avoid corrupting the latest checkpoint:

```text
write checkpoint.tmp
fsync checkpoint.tmp
rename checkpoint.tmp -> checkpoint.latest
write manifest.json
```

The exact implementation can vary, but the principle is stable: a crash should leave either the old valid checkpoint or the new valid checkpoint, not a half-written file.

## Practical Checks

- Save model weights, optimizer state, scheduler state, step count, and RNG seeds.
- Write atomically (temp file then rename) so a crash never corrupts the latest checkpoint.
- Keep a rolling window plus periodic milestones to bound disk use.
- Test resume on a small run before trusting it on a long one.
- Record code commit, environment, and dataset version alongside the checkpoint.
- During reconciliation, treat a checkpoint as resumable only after compatibility checks pass.
- Save mixed-precision scaler state when using fp16 training.
- For distributed training, save enough state to resume world size, sharding, sampler position, and rank-local state.
- Validate that the loaded checkpoint matches the current config and code expectations.
- Keep final artifacts separate from transient recovery checkpoints.

## Cadence

Checkpoint interval trades off overhead and lost work:

$$
\operatorname{expected\ lost\ work}
\approx
\frac{\Delta t_{\mathrm{ckpt}}}{2}
$$

where $\Delta t_{\mathrm{ckpt}}$ is the time between checkpoints. If checkpoints are too frequent, IO dominates. If they are too rare, preemption or timeout wastes too much compute.

## Completion Markers

For long jobs, a checkpoint is not the same as a completed output. Use an explicit marker or manifest:

$$
\text{complete}
\ne
\text{latest checkpoint exists}
$$

The manifest should identify the final step, expected shard count, config hash, and artifact type without exposing private paths.

## Compatibility Checks

Before resuming, compare:

- Config hash.
- Model architecture version.
- Dataset and split version.
- Optimizer and scheduler type.
- Precision mode.
- Distributed/sharding policy.
- Code commit or release identifier.

If these do not match, the run may load without crashing but continue as a different experiment.

## Related

- [[infra/hpc/slurm|Slurm]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]
- [[infra/hpc/preemption-resume|Preemption and resume]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[concepts/systems/distributed-training-runbook|Distributed training]]
- [[infra/index|Infra]]
