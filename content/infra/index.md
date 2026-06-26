---
title: Infra
tags:
  - infra
---

# Infra

서버, GPU, HPC, research engineering 운영 지식을 정리하는 입구입니다. 세부 문서는 일반화된 운영 방법만 남기고, 실제 내부 시스템 정보는 공개하지 않습니다.

이 페이지는 한글 안내 페이지입니다. 링크된 세부 `infra/*` 문서는 영어 canonical wiki note로 유지합니다.

운영 글은 문제를 그대로 적기보다 공개 가능한 runbook으로 정제합니다. 증상, 원인 후보, 수집할 evidence, 안전한 조치, 예방책을 남기고 private host, account name, SSH connection detail, internal path, credential, private queue name, user list, firewall detail, unpublished run result는 쓰지 않습니다.

## Main Areas

- [[infra/hpc/index|HPC]]
- [[infra/gpu/index|GPU]]
- [[infra/inference/index|Inference]]
- [[infra/training/index|Training]]
- [[infra/io/index|Storage and IO]]
- [[infra/environments/index|Environments]]
- [[infra/reproducibility/index|Reproducibility]]
- [[infra/server-ops/index|Server operations]]

## How To Read

- GPU utilization, memory, and bottlenecks belong under [[infra/gpu/index|GPU]].
- Slurm, resource requests, job arrays, preemption, and checkpointing belong under [[infra/hpc/index|HPC]].
- Serving capacity, batching, and latency planning belong under [[infra/inference/index|Inference]].
- Distributed training belongs under [[infra/training/index|Training]] and links back to [[concepts/systems/distributed-training|Distributed training]].
- Storage and dataloading problems belong under [[infra/io/index|Storage and IO]].
- Environment modules, containers, and run records belong under [[infra/environments/index|Environments]] and [[infra/reproducibility/index|Reproducibility]].

## 관련 입구

- [[concepts/systems/index|AI systems]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[projects/index|Project index]]
- [[projects/hpc-research-workflows|HPC research workflows]]
- [[concepts/evaluation/index|Evaluation]]
