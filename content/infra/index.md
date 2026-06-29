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
| [HPC](/infra/hpc) | Slurm, job, array, distributed training, preemption, checkpointing |
| [GPU](/infra/gpu) | utilization, memory, bottleneck, profiling signal |
| [Storage and IO](/infra/io) | storage, dataloading, throughput, cache behavior |
| [Reproducibility](/infra/reproducibility) | run record, artifact, environment capture |
| [Server operations](/infra/server-ops) | public runbook과 operational failure pattern |
| [Admin Usage Patterns](/infra/server-ops/admin-usage-patterns) | 자주 쓰는 서버/HPC 운영 명령을 public-safe하게 분류 |
| [Operations Command Cookbook](/infra/server-ops/operations-command-cookbook) | network, disk IO, GPU Xid, auth logs, Slurm inspection, sanitized command patterns |

## 운영 노트 작성 방식

서버 관리에서 얻은 지식은 `Infra`에 넣되, raw command dump가 아니라 재사용 가능한 public runbook으로 바꿉니다.

| Raw material | Infra note form |
| --- | --- |
| 자주 치는 명령어 | [[infra/server-ops/admin-usage-patterns|Admin Usage Patterns]]의 routing table |
| 명령 예시 | [[infra/server-ops/operations-command-cookbook|Operations Command Cookbook]]의 sanitized pattern |
| Slurm 정책/제한 | [[infra/hpc/slurm-accounting-limits|Slurm Accounting and Limits]]의 placeholder policy |
| `sbatch` option | [[infra/hpc/slurm-job-script|Slurm Job Script]]의 resource contract |
| GPU/Xid/driver 문제 | [[infra/gpu/index|GPU]]의 fault class와 diagnosis flow |
| disk/network 문제 | [[infra/hardware/storage-network|Storage and network]]와 [[infra/io/index|Storage and IO]] |

## 이동 기준

| 질문 | 이동할 곳 |
| --- | --- |
| hardware speed mental model이 필요한가? | [Hardware](/infra/hardware), [Memory hierarchy](/infra/hardware/memory-hierarchy) |
| GPU memory 또는 utilization 문제인가? | [GPU](/infra/gpu) |
| Slurm job lifecycle 또는 resource request인가? | [HPC](/infra/hpc), [Slurm](/infra/hpc/slurm) |
| Slurm account, fair-share, QOS, usage report인가? | [Slurm Accounting and Limits](/infra/hpc/slurm-accounting-limits) |
| 자주 쓰는 서버/HPC 운영 command를 정리하려는가? | [Admin Usage Patterns](/infra/server-ops/admin-usage-patterns), [Operations Command Cookbook](/infra/server-ops/operations-command-cookbook) |
| multi-node 또는 multi-GPU distributed run인가? | [Distributed training on HPC](/infra/hpc/distributed-training) |
| serving, batching, latency, inference contract인가? | [AI systems](/ai/systems), [Model serving](/concepts/systems/model-serving), [Latency and throughput](/concepts/systems/latency-throughput) |
| dataloading 또는 storage throughput 문제인가? | [Storage and IO](/infra/io) |
| environment, module, container 문제인가? | [Reproducibility](/infra/reproducibility), [Environment management](/concepts/systems/environment-management) |
| reproducible record가 필요한가? | [Reproducibility](/infra/reproducibility), [Run artifact](/concepts/systems/run-artifact) |

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
| Hardware / GPU | memory hierarchy, VRAM, utilization, driver/runtime symptom | architecture choice는 [AI](/ai/architectures) |
| HPC | Slurm job lifecycle, resource request, preemption, checkpoint survival | training objective와 optimizer는 [Machine Learning](/ai/machine-learning) |
| Storage and IO | filesystem throughput, dataloading stall, cache behavior | dataset semantics와 split design은 [Data concepts](/concepts/data) |
| Reproducibility | public-safe run record, artifact boundary, environment capture | statistical claim evidence는 [Evaluation](/ai/evaluation) |
| Server operations | access boundary, account, monitoring, backup, incident note | agent workflow design은 [Agents](/agents) |

## 관련 입구

| 영역 | 링크 |
| --- | --- |
| AI systems | [Systems](/ai/systems), [AI systems concepts](/concepts/systems) |
| Project workflow | [Projects](/projects), [HPC research workflows](/projects/hpc-research-workflows) |
| Evaluation boundary | [Evaluation](/concepts/evaluation) |
| Agent workflow | [LLM Wiki](/agents/workflows/llm-wiki) |
