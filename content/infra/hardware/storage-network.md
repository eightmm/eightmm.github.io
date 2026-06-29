---
title: Storage and Network
tags:
  - infra
  - hardware
  - storage
  - network
---

# Storage and Network

Storage and network determine how data enters a training job, how checkpoints survive, and how distributed workers communicate. A fast accelerator can still sit idle if the input path or interconnect is the true bottleneck.

## Storage Layers

| Layer | Good For | Risk |
| --- | --- | --- |
| CPU RAM cache | repeated reads during one run | disappears on process or node restart |
| local NVMe / SSD | hot dataset cache, temporary shards, checkpoint staging | not automatically shared or persistent |
| shared filesystem | common datasets, team artifacts, Slurm-visible paths | metadata contention and shared bandwidth |
| object storage / remote service | archival artifacts, cross-machine sharing | higher latency and request overhead |

## Network Layers

| Path | Use For | Watch |
| --- | --- | --- |
| node to shared storage | dataset reads, checkpoint writes | metadata storms, small files, contention |
| CPU host to GPU | batch transfer and pinned memory | copy time, synchronization |
| GPU to GPU in one node | data parallel, tensor parallel, peer transfer | topology and peer access |
| GPU to GPU across nodes | distributed training | all-reduce, bandwidth, latency, rank placement |
| client to service | inference serving | p95/p99 latency, queueing, payload size |

## Network Evidence

Realtime bandwidth tools should be selected by the operating environment. Public notes should describe the signal, not the private interface name or endpoint.

```bash
ip -s link show dev <interface>
sar -n DEV 1
ifstat <interface>
```

Use `netstat`/`ss` for socket-level symptoms, not as the only measure of interface bandwidth.

```bash
ss -s
netstat -s
```

Classify the result:

| Signal | Interpretation |
| --- | --- |
| high receive/transmit rate | bandwidth pressure or expected transfer |
| packet drops/errors | interface, driver, cable, switch, or overload class |
| retransmits | congestion or path-quality issue |
| many small connections | metadata, service, or client pattern issue |

## Training Data Path

The data path can be written as:

$$
\text{remote or shared storage}
\rightarrow
\text{local cache}
\rightarrow
\text{CPU preprocess}
\rightarrow
\text{pinned batch}
\rightarrow
\text{GPU VRAM}
$$

Each arrow can be the bottleneck.

## Distributed Training Path

For data parallel training, communication usually appears after backward:

$$
\text{forward}
\rightarrow
\text{backward}
\rightarrow
\text{gradient all-reduce}
\rightarrow
\text{optimizer step}
$$

If the all-reduce time grows faster than useful compute, adding more GPUs can reduce throughput per GPU.

## Practical Rules

- Cache hot datasets near the node when reuse is high and policy allows it.
- Avoid millions of tiny files when a shard format is acceptable.
- Separate checkpoint staging from long-term artifact storage.
- Use per-process IO tools to distinguish checkpoint writers, data loaders, copies, and archive extraction.
- Use controller health tools privately when disk failure is suspected, but publish only the health class.
- Measure dataloader time separately from GPU compute time.
- For distributed jobs, record world size, node count, GPU count per node, and interconnect assumption generically.
- Do not publish private mount paths, hostnames, live topology, device IDs, account names, or exact security controls.

## Storage Health Evidence

Disk performance symptoms and disk health symptoms are different. Slow training can come from storage contention even when hardware is healthy; a degraded array can also be masked until writes fail.

```bash
sudo iotop -o -a
sudo <raid-tool> /c<controller-id> show
```

| Evidence | Means |
| --- | --- |
| one process dominates write IO | checkpoint, logging, copy, or extraction bottleneck |
| many workers read tiny files | metadata contention or dataset layout problem |
| controller reports degraded array | hardware maintenance issue, not a model bug |
| rebuild in progress | temporary performance and reliability risk |

## Checks

| Question | Route |
| --- | --- |
| Is the issue dataloading? | [Data loading and IO](/infra/io/data-loading) |
| Is the issue checkpoint save/load? | [Checkpointing](/infra/hpc/checkpointing) |
| Is the issue distributed communication? | [Distributed training on HPC](/infra/hpc/distributed-training) |
| Is the issue serving latency? | [Inference capacity planning](/concepts/systems/inference-capacity-planning) |
| Is the issue a mount or permission problem? | [Storage mounts and permissions](/infra/server-ops/storage-mounts) |

## Related

- [[infra/hardware/index|Hardware]]
- [[infra/hardware/memory-hierarchy|Memory hierarchy]]
- [[infra/io/index|Storage and IO]]
- [[infra/hpc/distributed-training|Distributed training on HPC]]
- [[infra/server-ops/storage-mounts|Storage mounts and permissions]]
- [[concepts/systems/storage-io|Storage and IO]]
