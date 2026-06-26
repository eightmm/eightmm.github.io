---
title: Infra
tags:
  - infra
---

# Infra

서버, GPU, HPC, research engineering 운영 지식을 정리하는 입구입니다. 운영 경험은 특정 장비나 내부 환경보다 재사용 가능한 진단 절차와 runbook 형태로 남깁니다.

## Main Areas

| Area | Use For |
| --- | --- |
| [[infra/hpc/index|HPC]] | Slurm, jobs, arrays, preemption, checkpointing |
| [[infra/gpu/index|GPU]] | utilization, memory, bottlenecks, profiling signals |
| [[infra/inference/index|Inference]] | serving, batching, latency, throughput |
| [[infra/training/index|Training]] | distributed training, checkpoint state, training runs |
| [[infra/io/index|Storage and IO]] | storage, dataloading, throughput, cache behavior |
| [[infra/environments/index|Environments]] | modules, containers, package environments |
| [[infra/reproducibility/index|Reproducibility]] | run records, artifacts, environment capture |
| [[infra/server-ops/index|Server operations]] | public runbooks and operational failure patterns |

## Routing

| Question | Go To |
| --- | --- |
| GPU memory or utilization issue? | [[infra/gpu/index|GPU]] |
| Slurm job lifecycle or resource request? | [[infra/hpc/index|HPC]] |
| Serving capacity or latency planning? | [[infra/inference/index|Inference]] |
| Multi-GPU training behavior? | [[infra/training/index|Training]], [[concepts/systems/distributed-training|Distributed training]] |
| Dataloading or storage throughput issue? | [[infra/io/index|Storage and IO]] |
| Environment, module, or container problem? | [[infra/environments/index|Environments]] |
| Need a reproducible record? | [[infra/reproducibility/index|Reproducibility]] |

## 관련 입구

| Area | Link |
| --- | --- |
| AI systems concepts | [[concepts/systems/index|AI systems]] |
| Project workflow | [[projects/index|Projects]], [[projects/hpc-research-workflows|HPC research workflows]] |
| Evaluation boundary | [[concepts/evaluation/index|Evaluation]] |
| Agent workflow | [[agents/workflows/llm-wiki|LLM Wiki]] |
