---
title: Storage and IO
tags:
  - infra
  - storage
  - data-loading
---

# Storage and IO

Storage and IO note는 dataset, preprocessing, shared storage, local cache, accelerator input pipeline 사이의 경로를 다룹니다.

Data path는 system performance의 일부입니다.

$$
\text{dataset}
\rightarrow
\text{preprocess}
\rightarrow
\text{cache}
\rightarrow
\text{batch}
\rightarrow
\text{device}
$$

실제 limit이 file open latency, preprocessing CPU time, decompression, shared storage bandwidth, host-to-device transfer인데도 training job이 GPU 문제처럼 보일 수 있습니다.

## Scope

- Dataset access patterns and dataloader throughput.
- Cache design, feature materialization, and storage locality.
- Shared storage symptoms written as public runbooks.
- IO effects on GPU utilization and training reproducibility.

## IO Pattern Map

| Pattern | Symptom | Fix Direction |
| --- | --- | --- |
| Many small files | high metadata latency, low throughput | pack files, shard dataset, local cache |
| Heavy preprocessing | CPU-bound dataloader | precompute features, increase workers carefully |
| Remote shared storage | bursty stalls | stage to local scratch, reduce random access |
| Large sequential reads | bandwidth-bound | streaming format, compression tradeoff |
| Repeated feature extraction | duplicated work | versioned cache key and materialization |

## Notes

- [[infra/io/data-loading|Data loading and IO]]
- [[infra/hardware/storage-network|Storage and network]]
- [[infra/server-ops/storage-mounts|Storage mounts and permissions]]

## Checks

- Is the bottleneck small-file metadata, streaming bandwidth, CPU preprocessing, decompression, or transfer?
- Does the cache key include preprocessing version, data version, and split?
- Is preprocessing fit only on training data when labels or statistics are involved?
- Are private mount paths, hostnames, and storage topology removed?
- Does the note distinguish a one-off operational incident from a reusable IO pattern?

## Where New Notes Go

- Dataloader and cache behavior go here.
- Data semantics and split contracts go under [[concepts/data/index|Data]].
- Filesystem permissions and mounts go under [[infra/server-ops/index|Server operations]].
- GPU starvation diagnosis links back to [[infra/gpu/index#bottleneck-taxonomy|GPU bottleneck taxonomy]].

## Related

- [[concepts/systems/storage-io|Storage and IO]]
- [[infra/hardware/storage-network|Storage and network]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[infra/gpu/index#bottleneck-taxonomy|GPU bottleneck taxonomy]]
