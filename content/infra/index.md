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
| [HPC](/infra/hpc) | Slurm, jobs, arrays, preemption, checkpointing |
| [GPU](/infra/gpu) | utilization, memory, bottlenecks, profiling signals |
| [Inference](/infra/inference) | serving, batching, latency, throughput |
| [Training](/infra/training) | distributed training, checkpoint state, training runs |
| [Storage and IO](/infra/io) | storage, dataloading, throughput, cache behavior |
| [Environments](/infra/environments) | modules, containers, package environments |
| [Reproducibility](/infra/reproducibility) | run records, artifacts, environment capture |
| [Server operations](/infra/server-ops) | public runbooks and operational failure patterns |

## Routing

| Question | Go To |
| --- | --- |
| GPU memory or utilization issue? | [GPU](/infra/gpu) |
| Slurm job lifecycle or resource request? | [HPC](/infra/hpc) |
| Serving capacity or latency planning? | [Inference](/infra/inference) |
| Multi-GPU training behavior? | [Training](/infra/training), [Distributed training](/concepts/systems/distributed-training) |
| Dataloading or storage throughput issue? | [Storage and IO](/infra/io) |
| Environment, module, or container problem? | [Environments](/infra/environments) |
| Need a reproducible record? | [Reproducibility](/infra/reproducibility) |

## 관련 입구

| Area | Link |
| --- | --- |
| AI systems concepts | [AI systems](/concepts/systems) |
| Project workflow | [Projects](/projects), [HPC research workflows](/projects/hpc-research-workflows) |
| Evaluation boundary | [Evaluation](/concepts/evaluation) |
| Agent workflow | [LLM Wiki](/agents/workflows/llm-wiki) |
