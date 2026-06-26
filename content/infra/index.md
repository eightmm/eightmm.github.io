---
title: Infra
tags:
  - infra
---

# Infra

서버, GPU, HPC, research engineering 운영 지식을 정리하는 입구입니다. 세부 문서는 일반화된 운영 방법만 남기고, 실제 내부 시스템 정보는 공개하지 않습니다.

운영 글은 문제를 그대로 적기보다 공개 가능한 runbook으로 정제합니다. 증상, 원인 후보, 수집할 evidence, 안전한 조치, 예방책을 남기고 private host, account name, SSH connection detail, internal path, credential, private queue name, user list, firewall detail, unpublished run result는 쓰지 않습니다.

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

- [[concepts/systems/index|AI systems]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[projects/index|Project index]]
- [[projects/hpc-research-workflows|HPC research workflows]]
- [[concepts/evaluation/index|Evaluation]]
