---
title: Distributed Training on HPC
aliases:
  - infra/hpc/distributed
tags:
  - infra
  - hpc
  - distributed-training
---

# Distributed Training on HPC

Distributed training on HPC combines model training, GPU topology, storage, network, and scheduler behavior. The model-side concept is [[concepts/systems/distributed-training|Distributed training]]; this page focuses on public cluster execution patterns.

The cluster view is:

$$
\text{job allocation}
\rightarrow
\text{node layout}
\rightarrow
\text{process launch}
\rightarrow
\text{communication}
\rightarrow
\text{checkpoint and resume}
$$

## What Belongs Here

| Topic | Route |
| --- | --- |
| Slurm resource shape | [Resource request](/infra/hpc/resource-request) |
| Generic Slurm script | [Slurm job script](/infra/hpc/slurm-job-script) |
| GPU memory and utilization | [GPU](/infra/gpu) |
| Dataloader and storage stalls | [Storage and IO](/infra/io) |
| Distributed algorithm concept | [Distributed training concept](/concepts/systems/distributed-training) |
| Single-node training checks | [Training infra](/infra/training) |

## Launch Contract

A public distributed training note should state the shape, not private cluster details:

$$
\text{world size}
=
\text{nodes}
\times
\text{processes per node}
$$

Record these fields generically:

| Field | Why It Matters |
| --- | --- |
| nodes | network communication boundary |
| GPUs per node | topology and intra-node bandwidth |
| processes per GPU | usually one process per GPU for DDP |
| effective batch size | changes optimization and scaling |
| precision | memory and tensor-core behavior |
| checkpoint interval | recovery cost and storage pressure |
| launch method | scheduler integration and environment propagation |

## Scaling Checks

| Symptom | Likely Cause |
| --- | --- |
| single GPU works, multi-GPU diverges | effective batch, learning rate, gradient sync, seed/sampler mismatch |
| more GPUs do not speed up | communication, dataloader, small batch, synchronization, scheduler layout |
| occasional hangs | rank failure, network timeout, checkpoint barrier, uneven dataloader |
| OOM only in distributed run | per-rank batch, gradient buckets, sharding, activation memory |
| checkpoint resume changes behavior | missing optimizer, scheduler, RNG, sampler, scaler, or sharding state |

## Safe Public Notes

Do not publish private scheduler partitions, account names, hostnames, node names, ports, usernames, internal paths, live utilization, or unpublished results. Use placeholders such as `gpu-node`, `<partition>`, `<account>`, `<num_nodes>`, and `/path/to/project` only when a command template needs them.

## Minimal Workflow

1. Verify a single-device run.
2. Verify a single-node multi-GPU run.
3. Verify multi-node launch with a tiny smoke test.
4. Measure step time, dataloader time, GPU utilization, memory, and communication overhead.
5. Add checkpoint/restart before long runs.
6. Reconcile job state before relaunching.

## Related

- [[infra/hpc/index|HPC]]
- [[infra/hpc/slurm|Slurm]]
- [[infra/hpc/slurm-job-script|Slurm job script]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]
- [[infra/training/distributed-training|Distributed training]]
- [[infra/hardware/storage-network|Storage and network]]
- [[infra/gpu/index|GPU]]
