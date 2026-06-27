---
title: Memory Hierarchy
tags:
  - infra
  - hardware
  - memory
---

# Memory Hierarchy

The memory hierarchy explains why the same algorithm can be fast or slow depending on where its working set lives. Small memory close to compute is fast. Large memory far from compute is slower.

Approximate public mental model:

$$
\text{latency}
:
\text{register}
<
\text{L1}
<
\text{L2}
<
\text{L3}
<
\text{RAM}
<
\text{VRAM access across PCIe}
<
\text{SSD}
<
\text{network storage}
$$

Exact values vary by CPU, GPU, memory generation, interconnect, workload, and access pattern. Use the table for order of magnitude, not as a benchmark claim.

## Latency and Capacity Ladder

| Layer | Typical Role | Rough Latency Class | Capacity Class | Main Failure Mode |
| --- | --- | --- | --- | --- |
| CPU register | current scalar/vector operands | sub-ns | bytes to KB per core | too little reuse in tight loops |
| L1 cache | hottest CPU data/instructions | about 1 ns | tens of KB per core | random access misses |
| L2 cache | per-core working set | few ns | hundreds of KB to MB per core | working set too large |
| L3 cache | shared CPU cache | tens of ns | MB to tens of MB | cross-core contention |
| CPU RAM / DRAM | host tensors, dataloader buffers, OS page cache | tens to hundreds of ns | GB to TB | bandwidth or capacity pressure |
| GPU SRAM / shared memory | on-chip tile reuse inside kernels | very low | KB to MB per SM/block | poor tiling or occupancy |
| GPU VRAM / HBM | model weights, activations, KV cache | hundreds of ns class, very high bandwidth | GB to tens of GB per GPU | OOM, bandwidth-bound kernels |
| NVMe SSD | local dataset cache, checkpoints | microseconds to milliseconds | TB | small-file metadata and queue depth |
| Network storage | shared datasets, artifacts, checkpoints | milliseconds class | TB to PB | contention, metadata, mount issues |
| Remote service | object store, API, database | milliseconds to seconds | elastic | network latency and request limits |

## Bandwidth Ladder

Latency is delay for one access. Bandwidth is how much data can move per second once transfer is underway.

| Path | Useful For | Bottleneck Pattern |
| --- | --- | --- |
| CPU cache to core | tight loops, preprocessing, tokenization | cache miss rate |
| DRAM to CPU | dataloading, decompression, preprocessing | memory bandwidth saturation |
| CPU RAM to GPU VRAM | host-device transfer | PCIe or interconnect copy time |
| GPU HBM to tensor cores | training/inference kernels | memory-bound kernels |
| GPU to GPU | distributed training, tensor parallelism | all-reduce or peer-copy overhead |
| SSD to CPU RAM | dataset cache, checkpoint load/save | small files, queue depth, compression |
| network storage to node | shared dataset or output | metadata storms, shared bandwidth |

## Why This Matters for AI

Modern AI workloads are often limited by data movement:

$$
t_{\mathrm{step}}
=
t_{\mathrm{load}}
+ t_{\mathrm{host}}
+ t_{\mathrm{transfer}}
+ t_{\mathrm{compute}}
+ t_{\mathrm{sync}}
+ t_{\mathrm{save}}
$$

If $t_{\mathrm{compute}}$ is small but the rest is large, adding a faster GPU may not help.

## Common Diagnoses

| Symptom | Check |
| --- | --- |
| CPU preprocessing is slow | cache locality, vectorization, batch size, process count |
| GPU is idle between batches | dataloader timing, CPU RAM, storage path, host-device copy |
| OOM during training | parameters, activations, gradients, optimizer state, temporary buffers |
| OOM during LLM inference | weights plus KV cache, context length, concurrency, precision |
| checkpoint save stalls job | local vs network storage, checkpoint size, save frequency |
| distributed training slows down | GPU-GPU interconnect, all-reduce size, batch size, topology |

## Checks

- What data must be reused in the innermost loop?
- Does the working set fit in cache, RAM, VRAM, or only disk?
- Is the workload latency-bound, bandwidth-bound, compute-bound, or capacity-bound?
- Is the measured bottleneck local to one node, or caused by shared network/storage?
- Is a cache, memory format, batch size, precision, or checkpoint interval the simplest fix?

## Related

- [[infra/hardware/index|Hardware]]
- [[infra/hardware/storage-network|Storage and network]]
- [[infra/gpu/index#memory|GPU memory]]
- [[infra/io/data-loading|Data loading and IO]]
- [[infra/inference/capacity-planning|Inference capacity planning]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
