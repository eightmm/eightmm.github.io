---
title: Hardware
tags:
  - infra
  - hardware
---

# Hardware

Hardware notes explain the physical resource ladder behind AI systems: CPU registers and caches, SRAM, RAM, GPU VRAM, storage, and network. The goal is not to memorize exact vendor numbers, but to know which layer is likely limiting a workload.

Most infra problems can be reduced to one question:

$$
\text{bottleneck}
\in
\{\text{compute}, \text{capacity}, \text{bandwidth}, \text{latency}, \text{IO}, \text{network}, \text{scheduler}\}
$$

## Hardware Map

| Area | Use For | Start |
| --- | --- | --- |
| Memory hierarchy | registers, L1/L2/L3 cache, SRAM, DRAM, VRAM, disk, network latency | [Memory hierarchy](/infra/hardware/memory-hierarchy) |
| Storage and network | local disk, shared filesystem, object storage, Ethernet, InfiniBand-style interconnect | [Storage and network](/infra/hardware/storage-network) |
| GPU systems | VRAM, HBM bandwidth, tensor compute, host-device transfer, interconnect | [GPU](/infra/gpu) |
| HPC execution | scheduler, allocation, Slurm, distributed jobs, checkpoint/restart | [HPC](/infra/hpc) |

## Performance Ladder

The useful mental model is a ladder from fastest/smallest to slowest/largest:

$$
\text{register}
\rightarrow
\text{L1/L2/L3 cache}
\rightarrow
\text{RAM or VRAM}
\rightarrow
\text{local SSD}
\rightarrow
\text{network storage}
\rightarrow
\text{remote service}
$$

Moving data down this ladder usually increases capacity but also increases latency. Training and inference performance depend on keeping the hot working set near the compute units.

## Read This First When

| Symptom | Likely Layer |
| --- | --- |
| GPU utilization is low | data loading, CPU preprocessing, synchronization, network, or tiny kernels |
| model does not fit | GPU VRAM, activation memory, optimizer state, KV cache |
| many workers do not speed up | storage metadata, CPU contention, network, all-reduce, scheduler layout |
| training stalls every few steps | checkpoint IO, logging, validation, dataloader refill |
| distributed run scales poorly | interconnect bandwidth/latency, batch size, gradient synchronization |
| inference latency tail is high | batching policy, KV cache pressure, queueing, network, cold load |

## Resource Axes

| Axis | Meaning | Typical Fix Direction |
| --- | --- | --- |
| Compute | arithmetic units are the limit | larger batches, fused kernels, better tensor shapes |
| Capacity | model or batch does not fit | smaller batch/context, checkpointing, sharding |
| Bandwidth | moving data dominates arithmetic | layout, precision, fusion, cache locality |
| Latency | many small waits dominate | batching, fewer round trips, local cache |
| IO | storage cannot feed compute | materialization, streaming format, local scratch |
| Network | remote transfer or all-reduce dominates | placement, fewer syncs, better sharding |

## Related

- [[infra/hardware/memory-hierarchy|Memory hierarchy]]
- [[infra/hardware/storage-network|Storage and network]]
- [[infra/gpu/index|GPU]]
- [[infra/hpc/index|HPC]]
- [[infra/io/index|Storage and IO]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
