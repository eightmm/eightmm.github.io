---
title: Distributed Training
tags:
  - systems
  - training
  - distributed
---

# Distributed Training

Distributed training uses multiple devices or machines to train a model. The goal can be larger model capacity, larger batch size, faster wall-clock time, or memory fit.

For data parallel training with $N$ workers, each worker computes a gradient on a local mini-batch:

$$
g_i
=
\nabla_\theta
\frac{1}{B_{\mathrm{local}}}
\sum_{j=1}^{B_{\mathrm{local}}}
\mathcal{L}(f_\theta(x_{ij}), y_{ij})
$$

The synchronized gradient is:

$$
g
=
\frac{1}{N}
\sum_{i=1}^{N}
g_i
$$

The effective batch size is:

$$
B_{\mathrm{eff}}
=
N
\cdot
B_{\mathrm{local}}
\cdot
A
$$

where $A$ is the number of gradient accumulation steps.

## Parallelism Types

- Data parallelism: replicate the model, shard data, synchronize gradients.
- Tensor or model parallelism: split layers or matrix operations across devices.
- Pipeline parallelism: split model stages and stream micro-batches through them.
- Sharded data parallelism: shard parameters, gradients, and optimizer state.
- Expert parallelism: route tokens or examples to different experts.

## Scaling Limit

Training speed is limited by compute, communication, memory, input pipeline, and scheduler overhead:

$$
T_{\mathrm{step}}
\approx
\max(T_{\mathrm{compute}}, T_{\mathrm{comm}})
+
T_{\mathrm{input}}
+
T_{\mathrm{sync}}
$$

Adding devices helps only when the added parallelism reduces the dominant term more than it adds communication and coordination overhead.

## Checks

- Does single-device training produce a sensible loss curve before scaling out?
- Is $B_{\mathrm{eff}}$ reported, including accumulation and world size?
- Are learning rate schedule and warmup adjusted for the effective batch size?
- Are gradient accumulation steps counted consistently across workers?
- Are gradients synchronized at the intended frequency?
- Does checkpoint state include distributed sampler, optimizer shards, and global step?
- Is the bottleneck compute, communication, input pipeline, memory, or scheduler queue time?

## Related

- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/resource-scheduling|Resource scheduling]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[infra/training/distributed-training|Distributed training runbook]]
- [[infra/gpu/memory|GPU memory]]
- [[infra/hpc/slurm|Slurm]]
