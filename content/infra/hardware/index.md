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

| Area | Use for | Start |
| --- | --- | --- |
| Memory hierarchy | register, L1/L2/L3 cache, SRAM, DRAM, VRAM, disk, network latency | [Memory hierarchy](/infra/hardware/memory-hierarchy) |
| Storage and network | local disk, shared filesystem, object storage, Ethernet, InfiniBand-style interconnect | [Storage and network](/infra/hardware/storage-network) |
| GPU systems | VRAM, HBM bandwidth, tensor compute, host-device transfer, interconnect | [GPU](/infra/gpu) |
| HPC execution | scheduler, allocation, Slurm, distributed job, checkpoint/restart | [HPC](/infra/hpc) |

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

| Symptom | Likely layer |
| --- | --- |
| GPU utilization이 낮음 | data loading, CPU preprocessing, synchronization, network, tiny kernel |
| model이 memory에 들어가지 않음 | GPU VRAM, activation memory, optimizer state, KV cache |
| worker를 늘려도 빨라지지 않음 | storage metadata, CPU contention, network, all-reduce, scheduler layout |
| training이 몇 step마다 멈춤 | checkpoint IO, logging, validation, dataloader refill |
| distributed run scaling이 나쁨 | interconnect bandwidth/latency, batch size, gradient synchronization |
| inference latency tail이 큼 | batching policy, KV cache pressure, queueing, network, cold load |

## Resource Axes

| Axis | Meaning | Typical fix direction |
| --- | --- | --- |
| Compute | arithmetic unit이 limit | larger batch, fused kernel, 더 좋은 tensor shape |
| Capacity | model 또는 batch가 들어가지 않음 | smaller batch/context, checkpointing, sharding |
| Bandwidth | data movement가 arithmetic을 지배 | layout, precision, fusion, cache locality |
| Latency | 많은 small wait가 지배 | batching, fewer round trip, local cache |
| IO | storage가 compute를 먹이지 못함 | materialization, streaming format, local scratch |
| Network | remote transfer 또는 all-reduce가 지배 | placement, fewer sync, better sharding |

## Related

- [[infra/hardware/memory-hierarchy|Memory hierarchy]]
- [[infra/hardware/storage-network|Storage and network]]
- [[infra/gpu/index|GPU]]
- [[infra/hpc/index|HPC]]
- [[infra/io/index|Storage and IO]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
