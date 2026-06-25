---
title: AI
tags:
  - ai
---

# AI

AI 전반을 정리하는 입구입니다. 이 페이지는 공개 블로그 표면에 가까운 안내 페이지이고, 세부 개념은 영어 wiki 노트로 연결합니다.

이 페이지는 한글 안내 페이지입니다. 링크된 `concepts/*` 문서는 재사용 가능한 canonical wiki note로 영어를 유지합니다.

여기서 다루려는 핵심은 특정 모델 이름을 외우는 것이 아니라, 모델이 어떤 구조로 정보를 처리하고, 어떤 학습 신호로 표현을 만들고, 어떤 방식으로 생성하거나 판단하는지입니다.

## 큰 축

- Machine Learning: 예측 문제, feature, loss, regularization, validation을 다루는 기본 층
- Architecture: 모델이 정보를 흘려보내는 구조
- Learning: 어떤 supervision이나 objective로 표현을 학습하는지
- Generation: 데이터를 만들거나 변환하는 방식
- Evaluation: 모델이 실제로 일반화했는지 확인하는 방식

## Machine Learning

Machine learning은 AI 노트의 기본 층입니다. 딥러닝 모델을 보기 전에도 problem type, feature, loss, optimization, regularization, validation을 구분해야 합니다.

- [[ai/machine-learning|Machine learning gateway]]
- [[concepts/machine-learning/index|Machine learning]]
- [[concepts/machine-learning/linear-model|Linear model]]
- [[concepts/machine-learning/tree-based-model|Tree-based model]]
- [[concepts/machine-learning/kernel-method|Kernel method]]
- [[concepts/machine-learning/regularization|Regularization]]

## Architectures

아키텍처 노트는 입력의 형태와 inductive bias를 기준으로 봅니다. 이미지, sequence, graph, structure, set처럼 데이터가 달라지면 좋은 기본 구조도 달라집니다.

- [[ai/architectures|Architecture gateway]]
- [[concepts/architectures/index|Architectures]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/state-space-model|State-space models]]
- [[concepts/architectures/mamba|Mamba]]
- [[concepts/architectures/mixture-of-experts|Mixture of experts]]

## Learning Methods

학습 방법은 label이 충분한 상황과 부족한 상황을 나눠서 봅니다. 특히 [[concepts/learning/self-supervised-learning|self-supervised learning]], [[concepts/learning/jepa|JEPA]], [[concepts/learning/contrastive-learning|contrastive learning]]은 representation을 어떻게 만들 것인가와 직접 연결됩니다.

- [[ai/learning-methods|Learning methods gateway]]
- [[concepts/learning/index|Learning methods]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/preference-optimization|Preference optimization]]

## Generative Models

생성 모델은 likelihood, denoising, flow, autoregressive factorization처럼 서로 다른 관점에서 볼 수 있습니다. Bio-AI에서는 molecule generation, protein design, structure generation과 연결됩니다.

- [[ai/generative-models|Generative models gateway]]
- [[concepts/generative-models/index|Generative models]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]

## Evaluation

AI 노트는 평가 기준 없이 모델 목록이 되는 것을 피해야 합니다. 그래서 leakage, split, calibration, out-of-distribution generalization 같은 평가 노트를 계속 연결합니다.

- [[ai/evaluation|Evaluation gateway]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/calibration|Calibration]]

## Related

- [[bio-ai/index|Bio-AI]]
- [[agents/index|Agents]]
- [[ai/generative-models|Generative models]]
- [[ai/learning-methods|Learning methods]]
