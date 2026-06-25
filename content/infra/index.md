---
title: Infra
tags:
  - infra
---

# Infra

서버, GPU, HPC, research engineering 운영 지식을 정리하는 입구입니다. 세부 문서는 일반화된 운영 방법만 남기고, 실제 내부 시스템 정보는 공개하지 않습니다.

이 페이지는 한글 안내 페이지입니다. 링크된 세부 `infra/*` 문서는 영어 canonical wiki note로 유지합니다.

공개하면 안 되는 정보: private host, account name, SSH connection detail, internal path, credential, private queue name, user list, firewall detail, unpublished run result.

## Compute

- [[infra/gpu|GPU]]
- [[infra/gpu-utilization|GPU utilization]]
- [[infra/gpu-memory|GPU memory]]
- [[infra/hpc/slurm|Slurm]]
- [[infra/hpc/slurm-job-script|Slurm job script]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/checkpointing|Checkpointing]]

## Training and Serving

- [[concepts/systems/index|AI systems]]
- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/environment-management|Environment management]]
- [[concepts/systems/storage-io|Storage and IO]]
- [[concepts/systems/observability|Observability]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[concepts/systems/inference|Inference]]
- [[concepts/systems/batch-online-inference|Batch and online inference]]
- [[concepts/systems/model-serving|Model serving]]
- [[infra/data-loading-io|Data loading and IO]]
- [[infra/distributed-training|Distributed training]]
- [[infra/environment-modules-containers|Environment modules and containers]]
- [[infra/inference-serving|Inference serving]]
- [[infra/reproducible-run-record|Reproducible run record]]

## Server Operations

- [[infra/server-ops/index|Server operations]]

## 관련 입구

- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[projects/index|Project index]]
- [[projects/hpc-research-workflows|HPC research workflows]]
- [[concepts/evaluation/index|Evaluation]]
