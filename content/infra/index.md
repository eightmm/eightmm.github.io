---
title: Infra
tags:
  - infra
---

# Infra

서버, 하드웨어, GPU, HPC, storage, 운영 절차처럼 실제 실행 환경에 가까운 지식을 정리하는 입구입니다. 모델 학습법이나 serving 개념 자체보다, 장비와 운영 경계에서 반복되는 진단 절차와 runbook을 남깁니다.

Infra는 "모델이 무엇을 하는가"보다 "모델과 데이터가 어떤 실행 환경에서 실패하거나 느려지는가"를 봅니다. Training run, inference contract, model serving 개념은 [[ai/systems|AI Systems]]에서 시작하고, GPU memory, Slurm, shared filesystem, backup, monitoring처럼 장비와 운영 조건이 핵심이면 여기서 다룹니다.

## 주요 영역

| 영역 | 용도 |
| --- | --- |
| [Hardware](/infra/hardware) | CPU cache, RAM, VRAM, disk, network, performance hierarchy |
| [HPC](/infra/hpc) | Slurm, jobs, arrays, distributed training, preemption, checkpointing |
| [GPU](/infra/gpu) | utilization, memory, bottlenecks, profiling signals |
| [Storage and IO](/infra/io) | storage, dataloading, throughput, cache behavior |
| [Reproducibility](/infra/reproducibility) | run records, artifacts, environment capture |
| [Server operations](/infra/server-ops) | public runbooks and operational failure patterns |

## 이동 기준

| 질문 | 이동할 곳 |
| --- | --- |
| Need a hardware speed mental model? | [Hardware](/infra/hardware), [Memory hierarchy](/infra/hardware/memory-hierarchy) |
| GPU memory or utilization issue? | [GPU](/infra/gpu) |
| Slurm job lifecycle or resource request? | [HPC](/infra/hpc), [Slurm](/infra/hpc/slurm) |
| Multi-node or multi-GPU distributed run? | [Distributed training on HPC](/infra/hpc/distributed-training) |
| Serving, batching, latency, or inference contract? | [AI systems](/ai/systems), [Model serving](/concepts/systems/model-serving), [Latency and throughput](/concepts/systems/latency-throughput) |
| Dataloading or storage throughput issue? | [Storage and IO](/infra/io) |
| Environment, module, or container problem? | [Reproducibility](/infra/reproducibility), [Environment management](/concepts/systems/environment-management) |
| Need a reproducible record? | [Reproducibility](/infra/reproducibility), [Run artifact](/concepts/systems/run-artifact) |

## Infra 진단 흐름

문제가 생기면 바로 설정을 바꾸기보다 증거를 따라갑니다.

$$
\text{symptom}
\rightarrow
\text{resource axis}
\rightarrow
\text{minimal reproduction}
\rightarrow
\text{fix}
\rightarrow
\text{run record}
$$

| 단계 | 확인할 질문 |
| --- | --- |
| Symptom | OOM, slow step, queue wait, IO stall, failed import, lost artifact 중 무엇인가? |
| Resource axis | compute, memory, bandwidth, latency, storage, network, scheduler 중 무엇인가? |
| Reproduction | 짧고 공개 가능한 smoke run으로 재현되는가? |
| Fix boundary | 코드, 데이터, job request, environment, hardware 중 무엇을 바꾸는가? |
| Record | 같은 문제가 다시 왔을 때 볼 수 있는 public-safe note가 남았는가? |

## 경계

| 영역 | 여기 둘 것 | 다른 곳에서 다룰 것 |
| --- | --- | --- |
| Hardware / GPU | memory hierarchy, VRAM, utilization, driver/runtime symptoms | architecture choice goes to [AI](/ai/architectures) |
| HPC | Slurm job lifecycle, resource request, preemption, checkpoint survival | training objective and optimizer go to [Machine Learning](/ai/machine-learning) |
| Storage and IO | filesystem throughput, dataloading stalls, cache behavior | dataset semantics and split design go to [Data concepts](/concepts/data) |
| Reproducibility | public-safe run record, artifact boundary, environment capture | statistical claim evidence goes to [Evaluation](/ai/evaluation) |
| Server operations | access boundary, accounts, monitoring, backup, incident notes | agent workflow design goes to [Agents](/agents) |

## 관련 입구

| 영역 | 링크 |
| --- | --- |
| AI systems | [Systems](/ai/systems), [AI systems concepts](/concepts/systems) |
| Project workflow | [Projects](/projects), [HPC research workflows](/projects/hpc-research-workflows) |
| Evaluation boundary | [Evaluation](/concepts/evaluation) |
| Agent workflow | [LLM Wiki](/agents/workflows/llm-wiki) |
