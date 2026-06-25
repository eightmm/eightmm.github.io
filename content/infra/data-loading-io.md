---
title: Data Loading and IO
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
