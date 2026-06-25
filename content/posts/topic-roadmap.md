---
title: 글감 로드맵
tags:
  - posts
  - roadmap
---

# 글감 로드맵

이 페이지는 한글 post로 풀어낼 만한 글감을 모아두는 지도입니다. 세부 지식은 영어 wiki note로 이미 쌓고, post는 읽는 경로와 관점을 만들어 주는 역할을 합니다.

## AI 기본기

- Machine learning을 왜 [[concepts/data/index|Data]], [[concepts/tasks/index|Tasks]], [[concepts/architectures/index|Architectures]], [[concepts/evaluation/index|Evaluation]]로 나눠 봐야 하는가
- [[concepts/math/index|Math foundations]]를 AI 글에서 어느 정도까지 써야 하는가
- [[concepts/math/statistical-estimator|Statistical estimator]]와 [[concepts/math/bias-variance-tradeoff|bias-variance tradeoff]]를 실험 해석의 기본 언어로 쓰는 법
- [[concepts/evaluation/classification-metrics|Classification metrics]], [[concepts/evaluation/regression-metrics|regression metrics]], [[concepts/evaluation/generation-evaluation|generation evaluation]]을 task별로 구분하는 법
- Text, [[concepts/modalities/image|image]], [[concepts/modalities/video|video]], [[concepts/modalities/sequence|sequence]], [[concepts/modalities/3d-structure|3D structure]]를 modality 관점에서 비교하기
- [[concepts/modalities/modality-alignment|Modality alignment]]와 [[concepts/modalities/missing-modality|missing modality]]가 multimodal model 평가에서 왜 중요한가
- [[concepts/architectures/transformer|Transformer]], [[concepts/architectures/state-space-model|State-space model]], [[concepts/architectures/gnn|GNN]]을 입력 구조 관점에서 비교하기
- [[concepts/learning/self-supervised-learning|Self-supervised learning]], [[concepts/learning/contrastive-learning|Contrastive learning]], [[concepts/learning/jepa|JEPA]]를 representation 관점에서 정리하기

## Bio-AI

- [[bio-ai/index|Bio-AI]]를 구조기반/단백질/분자/유전체 입력으로 나눠 보는 이유
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]에서 pose generation, scoring, filtering을 분리해야 하는 이유
- [[concepts/sbdd/pose-quality|Pose quality]]와 [[concepts/sbdd/binding-affinity|Binding affinity]]를 혼동하면 생기는 문제
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]가 구조 기반 AI에서 중요한 이유
- [[concepts/genome-modeling/index|Genome modeling]]은 왜 이 블로그에서 넓은 omics가 아니라 sequence-level boundary topic으로 두는가

## Papers

- 논문 하나를 [[papers/paper-note-format|paper note]]에서 [[concepts/index|concept note]]로 분해하는 법
- [[papers/sbdd/posebusters|PoseBusters]]를 pose plausibility 체크리스트로 읽는 법
- [[papers/reading-status|Reading status]]를 써서 raw candidate와 verified note를 구분하는 법

## Infra

- [[infra/hpc/slurm|Slurm]]을 연구 workflow 관점에서 이해하기
- [[infra/gpu-memory|GPU memory]] 문제를 parameters, activations, optimizer state, KV cache로 나눠 보는 법
- [[infra/reproducible-run-record|Reproducible run record]]를 왜 연구 로그와 연결해야 하는가

## Agents

- [[agents/core/agent-architecture|Agent architecture]]를 model, state, tools, memory, verifier로 나눠 보는 법
- [[agents/tools/tool-contract|Tool contract]]가 agent 안정성에 중요한 이유
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]로 논문 후보를 모으고 검증하는 흐름
- [[agents/verification/verification-loop|Verification loop]]가 agent output보다 중요한 이유

## Writing Queue

- 구조 기반 AI 전체 지도
- Protein representation 입문
- Flow matching과 diffusion의 차이
- GNN과 equivariant GNN 비교
- Slurm job lifecycle 공개 버전
- Coding agent를 연구 블로그 운영에 쓰는 방식

## Related

- [[posts/index|Posts]]
- [[posts/blog-writing-guide|블로그 글 작성 가이드]]
- [[concepts/index|Concepts]]
- [[research/index|Research]]
- [[papers/index|Papers]]
