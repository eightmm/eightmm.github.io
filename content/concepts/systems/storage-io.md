---
title: Storage and IO
tags:
  - systems
  - storage
  - data-loading
---

# Storage and IO

Storage and IO describe how data moves from disk or network storage into CPU memory and then accelerator memory. Many training jobs are bottlenecked by input pipelines rather than model computation.

End-to-end step time can be approximated as:

$$
t_{\mathrm{step}}
\approx
\max(t_{\mathrm{compute}}, t_{\mathrm{input}})
+ t_{\mathrm{sync}}
$$

If input time dominates, accelerators wait:

$$
\operatorname{idle\ fraction}
\approx
\frac{\max(0, t_{\mathrm{input}}-t_{\mathrm{compute}})}
{t_{\mathrm{step}}}
$$

## Key Ideas

- Small random reads stress metadata and latency; large sequential reads stress bandwidth.
- Network storage, local scratch, object storage, and memory caches have different failure modes.
- Preprocessing can shift the bottleneck from storage to CPU.
- Caching helps only when cache keys include data and preprocessing versions.
- Validation and inference should use the same data interpretation as training.

## Practical Checks

- Are files many small objects, large shards, archives, databases, or streaming records?
- Is the bottleneck metadata, bandwidth, decompression, CPU transform, or transfer to GPU?
- Are workers overloading shared storage?
- Is local scratch cleaned and sized for the job?
- Are cached features invalidated when raw data or preprocessing changes?

## Related

- [[infra/data-loading-io|Data loading and IO]]
- [[infra/server-ops/storage-mounts|Storage mounts and permissions]]
- [[concepts/data/data-versioning|Data versioning]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
