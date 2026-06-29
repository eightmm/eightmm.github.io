---
title: Resource Request
tags:
  - infra
  - hpc
  - slurm
---

# Resource Request

Resource request는 shared cluster에서 job이 필요로 하는 CPU, GPU, memory, wall time, storage, special constraint를 설명합니다. Job과 scheduler 사이의 contract입니다.

For a job $j$:

$$
r_j = (c_j, g_j, m_j, \tau_j)
$$

$c_j$는 CPU count, $g_j$는 GPU count, $m_j$는 memory, $\tau_j$는 requested wall time입니다.

Scheduler는 이를 placement problem으로 봅니다.

$$
\operatorname{place}(j)
\quad\text{only if}\quad
r_j \le R_{\mathrm{available}}
$$

Oversized request는 job 자체가 단순해도 더 오래 기다릴 수 있습니다. Undersized request는 fail, OOM, slow run으로 이어질 수 있습니다.

## Generic Slurm Fields

```bash
#SBATCH --cpus-per-task=<cpu-count>
#SBATCH --gres=gpu:<gpu-count>
#SBATCH --mem=<memory>
#SBATCH --time=<hh:mm:ss>
```

위 값은 placeholder입니다. Private partition, account, hostname, internal path, cluster-specific name을 공개하지 않습니다.

## Smoke run으로 sizing하기

Full run을 요청하기 전에 small run을 측정합니다.

$$
\tau_{\mathrm{full}}
\approx
\tau_{\mathrm{smoke}}
\times
\frac{N_{\mathrm{full}}}{N_{\mathrm{smoke}}}
\times
\alpha
$$

여기서 $\alpha>1$은 IO, checkpointing, startup, validation, throughput variance를 위한 safety margin입니다.

Memory도 비슷하게 추정할 수 있습니다.

$$
m_{\mathrm{request}}
=
\max_t m_{\mathrm{observed}}(t)
\times
\beta
$$

여기서 $\beta$는 safety margin입니다. 이 margin은 observed variability로 정당화해야 하며, unknown behavior를 숨기기 위해 쓰면 안 됩니다.

## Bottleneck Matching

Resource request는 bottleneck과 맞아야 합니다.

- CPU-bound: preprocessing, compression, data transform, 일부 feature extraction.
- GPU-bound: dense tensor training, inference, docking kernel, geometric model.
- Memory-bound: large batch, large graph, high-resolution structure, data join.
- IO-bound: many small files, remote storage, checkpoint storm, dataset sharding.
- Scheduler-bound: 너무 많은 tiny job 또는 oversized monolithic job.

Data loading이나 CPU preprocessing이 bottleneck이면 GPU를 더 요청해도 도움이 되지 않습니다.

## Array vs Monolith

Independent shard에는 job array를 우선합니다.

$$
W
=
\bigcup_{k=1}^{K} W_k,
\qquad
W_i \cap W_j = \varnothing
$$

이 방식은 failure를 작게 만들고, scheduling flexibility를 높이며, failed shard 하나를 rerun하는 비용을 줄입니다.

## 실전 heuristic

- full training 또는 screening job 전에 small smoke run으로 시작합니다.
- log나 monitoring이 memory pressure를 보여줄 때만 memory를 늘립니다.
- wall time은 guess가 아니라 measured iteration speed에 기반해 요청합니다.
- 가능하면 CPU-heavy preprocessing과 GPU-heavy training을 분리합니다.
- final public-safe resource shape를 [[infra/reproducibility/run-record|Reproducible run record]]에 기록합니다.
- GPU count, batch size, data-loader worker, checkpoint interval을 measured throughput과 일관되게 유지합니다.
- public writeup에서는 site-specific resource 이름 대신 generic resource class를 설명합니다.

## 확인할 것

- request가 CPU, GPU, memory, IO, time 중 bottleneck과 맞는가?
- job이 빠르게 schedule되기엔 너무 큰가?
- workload를 [[infra/hpc/job-array|job array]]로 나눌 수 있는가?
- job이 wall-time limit 전에 checkpoint하는가?
- public note에서 cluster-specific value를 제거했는가?
- request가 smoke run 또는 이전 measured run에서 나온 것인가?
- wall-time estimate가 validation, checkpointing, cleanup과 compatible한가?
- 특정 인프라 세부 정보 없이도 run record에서 request를 이해할 수 있는가?

## Related

- [[infra/hpc/slurm|Slurm]]
- [[concepts/systems/resource-scheduling|Resource scheduling]]
- [[infra/hpc/slurm-job-script|Slurm job script]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/job-array|Job array]]
- [[infra/gpu/index#utilization|GPU utilization]]
- [[infra/io/data-loading|Data loading and IO]]
- [[infra/gpu/index#memory|GPU memory]]
