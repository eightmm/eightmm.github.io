---
title: Infra
tags:
  - infra
---

# Infra

서버, 하드웨어, GPU, HPC, research engineering 운영 지식을 정리하는 입구입니다. 운영 경험은 특정 장비나 내부 환경보다 재사용 가능한 진단 절차와 runbook 형태로 남깁니다.

## Main Areas

| Area | Use For |
| --- | --- |
| [Hardware](/infra/hardware) | CPU cache, RAM, VRAM, disk, network, performance hierarchy |
| [HPC](/infra/hpc) | Slurm, jobs, arrays, distributed training, preemption, checkpointing |
| [GPU](/infra/gpu) | utilization, memory, bottlenecks, profiling signals |
| [Storage and IO](/infra/io) | storage, dataloading, throughput, cache behavior |
| [Training](/infra/training) | training loop stability, checkpoint state, single-node checks |
| [Inference](/infra/inference) | serving, batching, latency, throughput |
| [Environments](/infra/environments) | modules, containers, package environments |
| [Reproducibility](/infra/reproducibility) | run records, artifacts, environment capture |
| [Server operations](/infra/server-ops) | public runbooks and operational failure patterns |

## Routing

| Question | Go To |
| --- | --- |
| Need a hardware speed mental model? | [Hardware](/infra/hardware), [Memory hierarchy](/infra/hardware/memory-hierarchy) |
| GPU memory or utilization issue? | [GPU](/infra/gpu) |
| Slurm job lifecycle or resource request? | [HPC](/infra/hpc), [Slurm](/infra/hpc/slurm) |
| Multi-node or multi-GPU distributed run? | [Distributed training on HPC](/infra/hpc/distributed-training) |
| Serving capacity or latency planning? | [Inference](/infra/inference) |
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
