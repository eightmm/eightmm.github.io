---
title: AI
tags:
  - ai
---

# AI

AI 전반을 정리하는 입구입니다. 이 페이지는 공개 블로그 표면에 가까운 한글 안내이고, 세부 개념은 영어 canonical wiki note로 유지합니다.

여기서 목표는 모델 이름을 많이 나열하는 것이 아니라, AI 문헌과 구현을 읽을 때 반복해서 나오는 축을 안정적으로 분리하는 것입니다.

$$
\text{AI system}
=
\text{data}
+ \text{representation}
+ \text{architecture}
+ \text{objective}
+ \text{evaluation}
+ \text{runtime}
$$

## 먼저 볼 지도

- [[math/index|Math]]: probability, linear algebra, calculus, likelihood, information theory처럼 모든 AI 노트가 기대는 수식 기반
- [[ai/machine-learning|Machine Learning]]: prediction problem, feature, loss, optimization, regularization, validation을 다루는 기본 층
- [[ai/architectures|Architectures]]: MLP, CNN, RNN, Transformer, GNN, SSM/Mamba, MoE처럼 정보가 흐르는 구조
- [[ai/learning-methods|Learning Methods]]: supervised, self-supervised, contrastive, JEPA, fine-tuning, preference/RL-style objective
- [[ai/generative-models|Generative Models]]: autoregressive, VAE, GAN, diffusion, score-based model, flow matching, normalizing flow
- [[ai/evaluation|Evaluation]]: metric, split, leakage, calibration, OOD generalization, uncertainty, failure analysis
- [[agents/index|Agents]]: tool use, memory, planning, verification, orchestration은 별도 영역으로 관리

## 분류 기준

AI note는 아래 질문으로 위치를 정합니다.

| 질문 | 둘 곳 |
| --- | --- |
| 필요한 수학 정의인가? | [[math/index|Math]] 또는 [[concepts/math/index|Math foundations]] |
| 입력과 출력이 무엇인가? | [[concepts/modalities/index|Modalities]], [[concepts/tasks/index|Tasks]] |
| 예측 문제와 loss의 기본형인가? | [[ai/machine-learning|Machine Learning]] |
| 모델 내부 구조인가? | [[ai/architectures|Architectures]] |
| supervision/objective/transfer 방식인가? | [[ai/learning-methods|Learning Methods]] |
| sample을 만들거나 distribution을 모델링하는가? | [[ai/generative-models|Generative Models]] |
| 성능 claim을 어떻게 검증하는가? | [[ai/evaluation|Evaluation]] |
| 실행, serving, reproducibility 문제인가? | [[concepts/systems/index|Systems]] 또는 [[infra/index|Infra]] |
| LLM이 도구를 쓰고 작업을 끝내는 방식인가? | [[agents/index|Agents]] |

## 기본 읽기 경로

1. [[math/index|Math]]에서 vector, probability, likelihood, entropy/KL, calculus를 먼저 잡습니다.
2. [[ai/machine-learning|Machine Learning]]에서 data, target, loss, optimization, validation의 기본 구조를 봅니다.
3. [[ai/architectures|Architectures]]에서 입력 구조별 inductive bias를 비교합니다.
4. [[ai/learning-methods|Learning Methods]]에서 label, pretraining signal, transfer, preference objective를 분리합니다.
5. [[ai/generative-models|Generative Models]]에서 likelihood, denoising, score, velocity, sampling 관점을 비교합니다.
6. [[ai/evaluation|Evaluation]]에서 split, metric, leakage, calibration, failure mode를 확인합니다.

## 입력 대상별 경로

- Text/sequence: [[concepts/modalities/text|Text]], [[concepts/modalities/sequence|Sequence]], [[concepts/architectures/transformer|Transformer]], [[concepts/architectures/state-space-model|State-space model]]
- Image/video: [[concepts/modalities/image|Image]], [[concepts/modalities/video|Video]], [[concepts/architectures/cnn|CNN]], [[concepts/architectures/vision-transformer|Vision Transformer]]
- Graph/set: [[concepts/modalities/graph|Graph]], [[concepts/architectures/gnn|Graph neural networks]], [[concepts/architectures/deep-sets|Deep Sets]]
- 3D/geometry: [[concepts/modalities/3d-structure|3D structure]], [[concepts/math/geometry|Geometry]], [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- Molecule/protein: [[bio-ai/index|Bio-AI]], [[concepts/molecular-modeling/index|Molecular modeling]], [[concepts/protein-modeling/index|Protein modeling]], [[concepts/sbdd/index|SBDD concepts]]
- Agent workflow: [[agents/core/index|Agent core]], [[agents/tools/index|Agent tools]], [[agents/verification/index|Agent verification]], [[agents/workflows/index|Agent workflows]]

## 작성 원칙

- 공개 표면은 한글로 간결하게 씁니다.
- 재사용 가능한 wiki note는 영어로 씁니다.
- 수식이 이해를 고정해 주는 경우에는 가능한 한 넣습니다.
- 모델 설명은 `input -> representation -> architecture -> objective -> evaluation` 순서로 정리합니다.
- 연구/논문/프로젝트로 커질 내용은 각각 [[research/index|Research]], [[papers/index|Papers]], [[projects/index|Projects]]로 보냅니다.

## Related

- [[concepts/index|Concepts]]
- [[math/index|Math]]
- [[bio-ai/index|Bio-AI]]
- [[infra/index|Infra]]
- [[agents/index|Agents]]
