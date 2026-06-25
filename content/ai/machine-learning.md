---
title: Machine Learning
tags:
  - ai
  - machine-learning
---

# Machine Learning

Machine learning은 AI를 이해하기 위한 기본 문법입니다. 복잡한 deep learning model도 결국 data, feature, objective, optimization, evaluation의 조합으로 봐야 합니다.

## Problem Types

- Classification: discrete label을 예측합니다.
- Regression: continuous value를 예측합니다.
- Ranking: 후보들을 순서화합니다.
- Clustering: label 없이 비슷한 sample을 묶습니다.
- Density estimation: data distribution 자체를 모델링합니다.
- Representation learning: downstream task에 쓸 embedding을 학습합니다.

## Model Families

- [[concepts/machine-learning/linear-model|Linear model]]
- [[concepts/machine-learning/tree-based-model|Tree-based model]]
- [[concepts/machine-learning/kernel-method|Kernel method]]
- [[concepts/machine-learning/ensemble-method|Ensemble method]]
- [[concepts/architectures/mlp|MLP]]

## Training Basics

- Objective: 무엇을 줄이거나 키울 것인가
- Loss: prediction error를 어떻게 수치화할 것인가
- [[concepts/machine-learning/optimization|Optimization]]: parameter를 어떻게 업데이트할 것인가
- [[concepts/machine-learning/regularization|Regularization]]: overfitting을 어떻게 줄일 것인가
- Validation: model selection을 어떤 split에서 할 것인가

## Representation and Features

Classical ML에서는 feature design이 중심이고, deep learning에서는 feature extractor 자체를 학습합니다. 이 차이를 이해하면 [[concepts/learning/self-supervised-learning|self-supervised learning]]이나 [[concepts/learning/transfer-learning|transfer learning]]이 왜 중요한지 더 분명해집니다.

- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/transfer-learning|Transfer learning]]

## Evaluation Questions

- train/test split이 deployment 상황을 반영하는가?
- metric이 실제 목표와 맞는가?
- baseline과 비교했을 때 개선이 의미 있는가?
- overfitting, leakage, distribution shift를 확인했는가?

## Related

- [[ai/learning-methods|Learning methods]]
- [[ai/evaluation|Evaluation]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
