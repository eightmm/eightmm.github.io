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
- [[concepts/machine-learning/empirical-risk-minimization|Empirical risk minimization]], [[concepts/machine-learning/stochastic-gradient|stochastic gradient]], [[concepts/machine-learning/gradient-descent|gradient descent]]를 training loop의 기본 수식으로 설명하기
- [[concepts/machine-learning/weight-decay|Weight decay]]와 [[concepts/machine-learning/gradient-clipping|gradient clipping]]을 optimizer 설정이 아니라 training stability와 generalization 관점에서 보기
- [[concepts/math/statistical-estimator|Statistical estimator]]와 [[concepts/math/bias-variance-tradeoff|bias-variance tradeoff]]를 실험 해석의 기본 언어로 쓰는 법
- [[concepts/evaluation/classification-metrics|Classification metrics]], [[concepts/evaluation/regression-metrics|regression metrics]], [[concepts/evaluation/generation-evaluation|generation evaluation]]을 task별로 구분하는 법
- [[concepts/evaluation/confusion-matrix|Confusion matrix]]와 [[concepts/evaluation/threshold-selection|threshold selection]]으로 classification 결과를 decision 관점에서 읽는 법
- [[concepts/evaluation/bootstrap-evaluation|Bootstrap evaluation]], [[concepts/evaluation/confidence-interval|confidence interval]], [[concepts/evaluation/reliability-diagram|reliability diagram]]으로 reported metric을 신뢰도 관점에서 해석하기
- [[concepts/data/example-unit|Example unit]], [[concepts/data/split-unit|split unit]], [[concepts/data/preprocessing-contract|preprocessing contract]]를 data leakage 방지의 기본 단어로 쓰는 법
- [[concepts/data/data-distribution|Data distribution]], [[concepts/data/data-schema|data schema]], [[concepts/data/label-semantics|label semantics]], [[concepts/data/dataset-shift|dataset shift]]를 모델보다 먼저 확인하는 이유
- [[concepts/architectures/inductive-bias|Inductive bias]], [[concepts/architectures/parameter-sharing|parameter sharing]], [[concepts/architectures/computational-complexity|computational complexity]]를 architecture 선택 기준으로 쓰는 법
- Text, [[concepts/modalities/image|image]], [[concepts/modalities/video|video]], [[concepts/modalities/sequence|sequence]], [[concepts/modalities/3d-structure|3D structure]]를 modality 관점에서 비교하기
- [[concepts/modalities/modality-alignment|Modality alignment]]와 [[concepts/modalities/missing-modality|missing modality]]가 multimodal model 평가에서 왜 중요한가
- [[concepts/architectures/transformer|Transformer]], [[concepts/architectures/state-space-model|State-space model]], [[concepts/architectures/gnn|GNN]]을 입력 구조 관점에서 비교하기
- [[concepts/learning/pretraining|Pretraining]], [[concepts/learning/fine-tuning|fine-tuning]], [[concepts/learning/instruction-tuning|instruction tuning]], [[concepts/learning/domain-adaptation|domain adaptation]]을 학습 pipeline 관점에서 나누기
- [[concepts/learning/self-supervised-learning|Self-supervised learning]], [[concepts/learning/contrastive-learning|Contrastive learning]], [[concepts/learning/jepa|JEPA]]를 representation 관점에서 정리하기
- [[concepts/llm/token-budget|Token budget]], [[concepts/llm/context-packing|context packing]], [[concepts/llm/tool-calling|tool calling]], [[concepts/llm/prompt-injection-boundary|prompt injection boundary]]를 LLM Wiki 운영 관점에서 묶어 설명하기

## Bio-AI

- [[bio-ai/index|Bio-AI]]를 구조기반/단백질/분자/유전체 입력으로 나눠 보는 이유
- [[entities/target|Target]], [[entities/assay|assay]], [[entities/dataset|dataset]]을 label이 생기는 context로 정리하기
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]], [[concepts/molecular-modeling/tautomer|tautomer]], [[concepts/molecular-modeling/protonation-state|protonation state]]가 split과 docking을 바꾸는 이유
- [[concepts/sbdd/receptor-ligand-preparation|Receptor and ligand preparation]]이 docking 결과를 좌우하는 이유
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]을 contact, pose, affinity, ranking으로 분해하는 법
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]에서 pose generation, scoring, filtering을 분리해야 하는 이유
- [[concepts/sbdd/template-leakage|Template leakage]]가 structure-based benchmark에서 왜 위험한가
- [[concepts/protein-modeling/pocket-representation|Pocket representation]]을 ligand-defined pocket과 deployable pocket으로 나눠 보는 법
- [[concepts/protein-modeling/sequence-identity-clustering|Sequence identity clustering]]을 protein-side split의 기본으로 쓰는 법
- [[concepts/molecular-modeling/molecular-property-prediction|Molecular property prediction]]에서 scaffold split, label semantics, activity cliff를 같이 봐야 하는 이유
- [[concepts/sbdd/pose-quality|Pose quality]]와 [[concepts/sbdd/binding-affinity|Binding affinity]]를 혼동하면 생기는 문제
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]가 구조 기반 AI에서 중요한 이유
- [[concepts/genome-modeling/index|Genome modeling]]은 왜 이 블로그에서 넓은 omics가 아니라 sequence-level boundary topic으로 두는가

## Papers

- 논문 하나를 [[papers/paper-note-format|paper note]]에서 [[concepts/index|concept note]]로 분해하는 법
- [[papers/claim-extraction|Claim extraction]], [[papers/evidence-table|evidence table]], [[papers/limitation-taxonomy|limitation taxonomy]]로 논문 주장을 좁히는 법
- [[papers/benchmark-card|Benchmark card]]와 [[papers/ablation-map|ablation map]]으로 benchmark와 component claim을 분리해서 읽는 법
- [[papers/reproduction-plan|Reproduction plan]]과 [[concepts/research-methodology/minimum-viable-experiment|minimum viable experiment]]로 논문을 실제 실험으로 줄이는 법
- [[papers/reproducibility-checklist|Reproducibility checklist]]로 논문 구현 가능성을 빠르게 판별하는 법
- [[concepts/research-methodology/threat-to-validity|Threat to validity]]를 이용해 paper claim을 좁히는 법
- [[concepts/research-methodology/literature-synthesis|Literature synthesis]]로 여러 논문을 하나의 연구 질문으로 묶는 법
- [[papers/sbdd/posebusters|PoseBusters]]를 pose plausibility 체크리스트로 읽는 법
- [[papers/reading-status|Reading status]]를 써서 raw candidate와 verified note를 구분하는 법

## Infra

- [[infra/hpc/slurm|Slurm]]을 연구 workflow 관점에서 이해하기
- [[concepts/systems/resource-scheduling|Resource scheduling]], [[infra/hpc/resource-request|resource request]], [[infra/hpc/job-array|job array]]를 shared HPC 사용의 기본 단어로 정리하기
- [[infra/gpu-memory|GPU memory]] 문제를 parameters, activations, optimizer state, KV cache로 나눠 보는 법
- [[concepts/systems/checkpoint-state|Checkpoint state]]와 [[concepts/systems/failure-recovery|failure recovery]]를 긴 training run의 기본 설계로 보는 법
- [[infra/hpc/preemption-resume|Preemption and resume]]를 긴 실험의 reliability 문제로 설명하기
- [[concepts/systems/batch-online-inference|Batch and online inference]]를 throughput/latency 관점에서 나눠 보는 법
- [[logs/public-incident-note|Public incident note]]로 운영 실패를 공개 가능한 lesson으로 바꾸는 법
- [[infra/reproducible-run-record|Reproducible run record]]를 왜 연구 로그와 연결해야 하는가

## Agents

- [[agents/core/agent-architecture|Agent architecture]]를 model, state, tools, memory, verifier로 나눠 보는 법
- [[agents/core/agent-environment|Agent environment]], [[agents/core/action-space|action space]], [[agents/tools/tool-result-handling|tool result handling]]을 agent loop의 기본 단어로 정리하기
- [[agents/verification/acceptance-criteria|Acceptance criteria]]로 agent가 "끝났다"고 말하기 전에 무엇을 증명해야 하는지 정리하기
- [[agents/core/task-decomposition|Task decomposition]]과 [[agents/workflows/agent-handoff|agent handoff]]로 긴 작업을 끊어 넘기는 법
- [[agents/workflows/agent-runbook|Agent runbook]]으로 반복되는 논문/블로그/wiki 작업을 표준화하는 법
- [[inbox/curation-queue|Curation queue]]와 [[inbox/publishing-gate|publishing gate]]로 agent output을 공개 note로 승격하는 법
- [[agents/tools/tool-contract|Tool contract]]가 agent 안정성에 중요한 이유
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]로 논문 후보를 모으고 검증하는 흐름
- [[agents/verification/verification-loop|Verification loop]]가 agent output보다 중요한 이유

## Writing Queue

- 구조 기반 AI 전체 지도
- Protein representation 입문
- Flow matching과 diffusion의 차이
- GNN과 equivariant GNN 비교
- Negative result를 [[concepts/research-methodology/negative-result|public research lesson]]으로 정리하는 방식
- Slurm job lifecycle 공개 버전
- Coding agent를 연구 블로그 운영에 쓰는 방식

## Related

- [[posts/index|Posts]]
- [[posts/blog-writing-guide|블로그 글 작성 가이드]]
- [[concepts/index|Concepts]]
- [[research/index|Research]]
- [[papers/index|Papers]]
