---
title: Distributed Training Runbook
aliases:
  - infra/distributed-training
tags:
  - systems
  - training
  - distributed
---

# Distributed Training Runbook

Distributed training splits a model or its data across multiple devices to fit larger models or finish faster. The conceptual layer is [[concepts/systems/distributed-training|Distributed training]]; this note focuses on public operational checks.

For scheduler-managed multi-node jobs, use [[infra/hpc/distributed-training|Distributed training on HPC]]. This page remains a compact training-side checklist.

The first question is not "how many GPUs can I use?" but which limit the single-device run already hit.

$$
\text{scale decision}
=
f(\text{memory fit}, \text{step time}, \text{input pipeline}, \text{communication}, \text{failure cost})
$$

## When To Scale

| Signal | Better First Move | Scale-Out Move |
| --- | --- | --- |
| model does not fit | activation checkpointing, mixed precision, smaller batch | sharded data parallel or model parallel |
| step is compute-bound | larger local batch, fused kernels, precision check | data parallel if communication remains small |
| GPU waits for data | dataloader, cache, shard format, storage path | do not add GPUs yet |
| validation needs many independent runs | job array or multi-run scheduler plan | distributed training may be unnecessary |
| walltime risk is high | checkpoint/resume and shorter jobs | distributed only if recovery is defined |

## Batch and Step Contract

Record the batch arithmetic before interpreting a learning curve:

$$
B_{\mathrm{eff}}
=
N_{\mathrm{workers}}
\cdot B_{\mathrm{local}}
\cdot A
$$

where $A$ is gradient accumulation. If $B_{\mathrm{eff}}$ changes, the optimizer step, learning-rate schedule, warmup, gradient clipping, and number of consumed examples per step may change too.

The training log should make these counters unambiguous:

| Counter | Meaning |
| --- | --- |
| micro-step | one forward/backward on a local micro-batch |
| optimizer step | one parameter update after accumulation and synchronization |
| global step | shared step index used by checkpoint, scheduler, and logging |
| consumed samples | examples processed across all workers |

## Single-GPU Gate

Do not debug distributed behavior before a small single-device run is sane.

| Check | Evidence |
| --- | --- |
| loss decreases on a small run | short training curve |
| data order is deterministic enough | seed and sampler policy |
| checkpoint can save and resume | state dict, optimizer, scheduler, scaler, step |
| memory is understood | weight, activation, optimizer, batch contribution |
| dataloader keeps up | dataloader wait or step timing |

## Communication Gate

For data parallel training, useful scaling depends on communication not dominating compute:

$$
\eta_{\mathrm{scaling}}
=
\frac{T_1}{N \cdot T_N}
$$

where $T_1$ is single-device step time and $T_N$ is step time with $N$ workers. Low efficiency can be a valid tradeoff, but the note should say why.

| Symptom | Likely Cause |
| --- | --- |
| speed barely improves | all-reduce, input pipeline, sync logging, small batch |
| loss changes after scaling | effective batch, sampler, LR schedule, batch norm, nondeterminism |
| hang at startup | launcher, rank environment, rendezvous setting, filesystem, NCCL setup |
| OOM only after scaling | gradient buckets, shards, per-rank batch, validation batch |

## Practical Checks

- Start with data parallelism (DDP); reach for model/pipeline parallel only when memory forces it.
- Verify gradients are synchronized and the effective batch size is what you intend.
- Watch communication overhead — interconnect bandwidth often caps scaling.
- Confirm a single-GPU run matches loss curves before scaling out.
- Checkpoint regularly so a failed node does not lose a long run.
- Use [[infra/gpu/index#bottleneck-taxonomy|GPU bottleneck taxonomy]] to separate single-device bottlenecks from communication bottlenecks.

## Public Run Record Fields

| Field | Why |
| --- | --- |
| world size and local batch | reconstructs effective batch |
| accumulation steps | separates micro-step from optimizer step |
| precision and checkpointing | explains memory and stability |
| launcher and parallelism type | distinguishes DDP, sharding, pipeline, model parallel |
| checkpoint contents | proves resume boundary |
| hardware class | enough context without publishing private node names |
| failure class | useful for future runs without exposing logs |

## Boundary

Keep this page about the training-side run contract. Move cluster-specific commands, partitions, node placement, and Slurm diagnostics to [[infra/hpc/distributed-training|Distributed training on HPC]]. Move hardware bottleneck details to [[infra/gpu/index|GPU]] and [[infra/hardware/storage-network|Storage and Network]].

## Related

- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/systems/distributed-training|Distributed training]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[infra/gpu/index|GPU]]
- [[infra/gpu/index#bottleneck-taxonomy|GPU bottleneck taxonomy]]
- [[infra/gpu/index#memory|GPU memory]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/hpc/distributed-training|Distributed training on HPC]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[infra/hpc/slurm|Slurm]]
- [[ai/systems|Systems]]
- [[infra/index|Infra]]
