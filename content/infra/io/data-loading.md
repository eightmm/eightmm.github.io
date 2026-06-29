---
title: Data Loading and IO
aliases:
  - infra/data-loading-io
tags:
  - infra
  - data
  - training
---

# Data Loading and IO

Data loading and storage I/O often determine whether GPUs stay busy. A slow input pipeline can make a correct training job look like a GPU performance problem.

Throughput can be viewed as:

$$
\operatorname{utilization}
\approx
\frac{t_{\mathrm{compute}}}
{t_{\mathrm{compute}} + t_{\mathrm{input}}}
$$

If $t_{\mathrm{input}}$ is large, GPU utilization falls even when the model code is efficient.

## Checks

- Is the bottleneck file open latency, network storage bandwidth, CPU preprocessing, decompression, or host-to-device transfer?
- Are small files being read one by one from shared storage?
- Are dataloader workers saturating CPU or fighting over I/O?
- Can preprocessing be cached with a versioned feature key?
- Does validation reuse the same preprocessing path as training?

## Bottleneck Taxonomy

| Bottleneck | Symptom | Typical fix |
| --- | --- | --- |
| Small-file latency | many workers, low bandwidth, high metadata pressure | shard files, use tar/LMDB/WebDataset-style packing, cache features |
| Network storage | GPU waits while storage throughput is saturated | stage to scratch, reduce repeated reads, prefetch carefully |
| CPU preprocessing | high CPU usage, low GPU utilization | vectorize, cache deterministic transforms, move cheap transforms to GPU |
| Decompression | workers busy but batches arrive slowly | choose faster codecs, predecode stable data, increase worker memory carefully |
| Host-to-device copy | batches ready on CPU but GPU step waits | use pinned memory, overlap copy with compute, avoid excessive object nesting |
| Validation path drift | train fast but eval slow or inconsistent | share preprocessing contract, version cached features |

## Measurement Pattern

Record time by stage before changing the model:

$$
T_{\mathrm{batch}}
=
T_{\mathrm{read}}
+ T_{\mathrm{decode}}
+ T_{\mathrm{transform}}
+ T_{\mathrm{collate}}
+ T_{\mathrm{host\rightarrow device}}
+ T_{\mathrm{compute}}
$$

If only total step time is recorded, an I/O problem can be mistaken for optimizer, model, or GPU inefficiency. A public run note should record the symptom, the measurement method, and the mitigation, not live storage paths or machine-specific topology.

## Practical Rules

- Prefer fewer larger shards over millions of tiny files on shared storage.
- Cache deterministic preprocessing with an explicit version key.
- Keep train, validation, and inference preprocessing contracts comparable.
- Avoid over-increasing worker count when the shared filesystem is already saturated.
- Treat low GPU utilization as a symptom, not a diagnosis.

## Public Notes

- Use generic storage names such as `shared-storage` and `scratch`.
- Do not publish private mount paths or topology.
- Record the symptom and mitigation, not live filesystem details.

## Related

- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/systems/storage-io|Storage and IO]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/server-ops/storage-mounts|Storage mounts and permissions]]
- [[concepts/evaluation/leakage|Leakage]]
