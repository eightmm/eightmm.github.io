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

- dataset access pattern과 dataloader throughput.
- cache design, feature materialization, storage locality.
- public runbook으로 쓸 수 있는 shared storage symptom.
- GPU utilization과 training reproducibility에 미치는 IO effect.

## IO Pattern Map

| Pattern | Symptom | Fix direction |
| --- | --- | --- |
| Many small files | metadata latency가 높고 throughput이 낮음 | file packing, dataset sharding, local cache |
| Heavy preprocessing | CPU-bound dataloader | feature precompute, worker 수를 조심스럽게 증가 |
| Remote shared storage | bursty stall | local scratch staging, random access 감소 |
| Large sequential reads | bandwidth-bound | streaming format, compression tradeoff |
| Repeated feature extraction | duplicated work | versioned cache key와 materialization |

## Notes

- [[infra/io/data-loading|Data loading and IO]]
- [[infra/hardware/storage-network|Storage and network]]
- [[infra/server-ops/storage-mounts|Storage mounts and permissions]]

## Checks

- bottleneck이 small-file metadata, streaming bandwidth, CPU preprocessing, decompression, transfer 중 무엇인가?
- cache key가 preprocessing version, data version, split을 포함하는가?
- label 또는 statistic이 관련된 preprocessing은 training data에만 fit되는가?
- private mount path, hostname, storage topology를 제거했는가?
- note가 one-off operational incident와 reusable IO pattern을 구분하는가?

## Routing

| Question | Go To |
| --- | --- |
| Is this dataloader throughput, cache behavior, or storage locality? | [Data loading and IO](/infra/io/data-loading), [Storage and IO](/concepts/systems/storage-io) |
| Is this data semantics, preprocessing, or split policy? | [Data](/concepts/data), [Preprocessing contract](/concepts/data/preprocessing-contract) |
| Is this filesystem permission or mount behavior? | [Server operations](/infra/server-ops), [Storage mounts and permissions](/infra/server-ops/storage-mounts) |
| Is the visible symptom low GPU utilization? | [GPU bottleneck taxonomy](/infra/gpu#bottleneck-taxonomy) |

## Related

- [[concepts/systems/storage-io|Storage and IO]]
- [[infra/hardware/storage-network|Storage and network]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[infra/gpu/index#bottleneck-taxonomy|GPU bottleneck taxonomy]]
