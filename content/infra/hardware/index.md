---
title: Hardware
tags:
  - infra
  - hardware
---

# Hardware

Hardware note는 AI system 뒤의 physical resource ladder를 설명합니다. CPU register/cache, SRAM, RAM, GPU VRAM, storage, network가 여기에 들어갑니다. 목표는 vendor별 숫자를 외우는 것이 아니라 어떤 layer가 workload를 제한하는지 판단하는 것입니다.

대부분의 infra 문제는 하나의 질문으로 줄일 수 있습니다.

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

유용한 mental model은 가장 빠르고 작은 층에서 가장 느리고 큰 층으로 내려가는 ladder입니다.

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

Data를 이 ladder 아래로 내릴수록 capacity는 보통 커지지만 latency도 커집니다. Training과 inference performance는 hot working set을 compute unit 가까이에 유지할 수 있는지에 크게 좌우됩니다.

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
