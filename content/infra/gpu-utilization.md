---
title: GPU Utilization
tags:
  - infra
  - gpu
  - performance
---

# GPU Utilization

GPU utilization is a symptom, not a complete diagnosis. Low utilization can come from data loading, CPU preprocessing, synchronization, communication, small batch sizes, memory stalls, or simply a workload that is not compute-heavy.

A simple utilization view is:

$$
u
\approx
\frac{t_{\mathrm{gpu\ work}}}
{t_{\mathrm{wall}}}
$$

But high utilization alone does not prove good throughput. The useful question is:

$$
\text{work completed per second}
$$

under a fixed quality and memory budget.

## Diagnosis Pattern

1. Measure step time, GPU utilization, GPU memory, CPU usage, and I/O wait.
2. Check whether data loading keeps up with compute.
3. Check batch size, precision, and tensor shapes.
4. Check synchronization points and distributed communication.
5. Profile a short representative run before changing code.

## Practical Checks

- Is GPU memory full but utilization low?
- Are dataloader workers or storage saturated?
- Is the model too small to fill the GPU?
- Are there frequent CPU-GPU transfers or synchronization calls?
- Is the metric throughput, latency, cost, or time-to-train?
- Which bottleneck class best explains the utilization pattern?

## Public Notes

- Use generic symptoms and mitigation patterns.
- Do not publish live process lists, user names, hostnames, device serials, or private dashboards.

## Related

- [[infra/gpu|GPU]]
- [[infra/gpu-bottleneck-taxonomy|GPU bottleneck taxonomy]]
- [[infra/gpu-memory|GPU memory]]
- [[infra/data-loading-io|Data loading and IO]]
- [[infra/distributed-training|Distributed training]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[concepts/systems/storage-io|Storage and IO]]
