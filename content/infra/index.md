---
title: Infra
tags:
  - infra
---

# Infra

서버, 하드웨어, GPU, HPC, storage, 운영 절차처럼 실제 실행 환경에 가까운 지식을 정리하는 입구입니다. 모델 학습법이나 serving 개념 자체보다, 장비와 운영 경계에서 반복되는 진단 절차와 runbook을 남깁니다.

## Main Areas

| Area | Use For |
| --- | --- |
| [Hardware](/infra/hardware) | CPU cache, RAM, VRAM, disk, network, performance hierarchy |
| [HPC](/infra/hpc) | Slurm, jobs, arrays, distributed training, preemption, checkpointing |
| [GPU](/infra/gpu) | utilization, memory, bottlenecks, profiling signals |
| [Storage and IO](/infra/io) | storage, dataloading, throughput, cache behavior |
| [Reproducibility](/infra/reproducibility) | run records, artifacts, environment capture |
| [Server operations](/infra/server-ops) | public runbooks and operational failure patterns |

## Routing

| Question | Go To |
| --- | --- |
| Need a hardware speed mental model? | [Hardware](/infra/hardware), [Memory hierarchy](/infra/hardware/memory-hierarchy) |
| GPU memory or utilization issue? | [GPU](/infra/gpu) |
| Slurm job lifecycle or resource request? | [HPC](/infra/hpc), [Slurm](/infra/hpc/slurm) |
| Multi-node or multi-GPU distributed run? | [Distributed training on HPC](/infra/hpc/distributed-training) |
| Serving, batching, latency, or inference contract? | [AI systems](/concepts/systems), [Model serving](/concepts/systems/model-serving), [Latency and throughput](/concepts/systems/latency-throughput) |
| Dataloading or storage throughput issue? | [Storage and IO](/infra/io) |
| Environment, module, or container problem? | [Reproducibility](/infra/reproducibility), [Environment management](/concepts/systems/environment-management) |
| Need a reproducible record? | [Reproducibility](/infra/reproducibility), [Run artifact](/concepts/systems/run-artifact) |

## 관련 입구

| Area | Link |
| --- | --- |
| AI systems concepts | [AI systems](/concepts/systems) |
| Project workflow | [Projects](/projects), [HPC research workflows](/projects/hpc-research-workflows) |
| Evaluation boundary | [Evaluation](/concepts/evaluation) |
| Agent workflow | [LLM Wiki](/agents/workflows/llm-wiki) |
