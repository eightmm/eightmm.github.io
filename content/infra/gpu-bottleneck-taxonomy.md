---
title: GPU Bottleneck Taxonomy
tags:
  - infra
  - gpu
  - performance
---

# GPU Bottleneck Taxonomy

A GPU bottleneck taxonomy separates capacity, bandwidth, compute, communication, input pipeline, and scheduler limits. It prevents "GPU utilization is low" from becoming a vague diagnosis.

A training or inference step can be decomposed as:

$$
t_{\mathrm{step}}
=
t_{\mathrm{input}}
+ t_{\mathrm{host}}
+ t_{\mathrm{transfer}}
+ t_{\mathrm{gpu}}
+ t_{\mathrm{sync}}
+ t_{\mathrm{comm}}
$$

where input loading, CPU preprocessing, host-device transfer, GPU kernels, synchronization, and distributed communication can each dominate.

## Bottleneck Types

| Type | Symptom | Typical evidence |
| --- | --- | --- |
| Capacity | Out-of-memory or tiny batch size | Memory allocation, batch/context sensitivity |
| Bandwidth | High memory traffic, weak FLOP utilization | Profiler memory bandwidth, tensor shape pattern |
| Compute | Kernels dominate wall time | High GPU utilization and stable input pipeline |
| Input pipeline | GPU waits for data | CPU usage, data-loader timing, storage IO |
| Synchronization | Frequent stalls | Host sync calls, logging, metric aggregation |
| Communication | Poor multi-GPU scaling | All-reduce time, interconnect saturation |
| Scheduler | Long wait or preemption | Queue time, walltime, resource request mismatch |

## Diagnostic Order

1. Measure step time and throughput under a fixed workload.
2. Check [[infra/gpu-memory|GPU memory]] and batch/context sensitivity.
3. Check [[infra/gpu-utilization|GPU utilization]] together with CPU, IO, and data loading.
4. Profile a short representative run before changing architecture.
5. Separate single-GPU bottlenecks from distributed communication bottlenecks.
6. Record the public-safe conclusion in a log, project milestone, or infra note.

## Checks

- Is the bottleneck capacity, bandwidth, compute, input, synchronization, communication, or scheduler?
- Does the metric optimize time-to-train, samples/sec, tokens/sec, latency, throughput, or cost?
- Does a smaller batch, shorter context, or simpler preprocessing change the bottleneck?
- Are profiling results summarized without private hostnames, paths, usernames, or live metrics?

## Related

- [[infra/gpu|GPU]]
- [[infra/gpu-memory|GPU memory]]
- [[infra/gpu-utilization|GPU utilization]]
- [[infra/data-loading-io|Data loading and IO]]
- [[infra/distributed-training|Distributed training]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[concepts/systems/observability|Observability]]
