---
title: Machine Learning
tags:
  - ai
  - machine-learning
---

# Machine Learning

Machine learning은 AI를 이해하기 위한 기본 문법입니다. 복잡한 deep learning model도 결국 data, feature, objective, optimization, evaluation의 조합으로 봐야 합니다.

이 페이지는 한글 안내 페이지입니다. 링크된 `concepts/machine-learning/*` 문서는 영어 canonical wiki note로 유지합니다.

가장 기본적인 관점은 데이터셋 $\mathcal{D}$에서 모델 $f_\theta$를 학습해, 보지 못한 입력에서도 손실이 작아지게 만드는 것입니다.

$$
\hat{\theta}
= \arg\min_\theta \frac{1}{n}\sum_{i=1}^n
\mathcal{L}(f_\theta(x_i), y_i)
$$

여기서 $x_i$는 입력, $y_i$는 label 또는 target, $\mathcal{L}$은 loss function, $\theta$는 model parameter입니다.

## 문제 유형

- [[concepts/machine-learning/probabilistic-prediction|Probabilistic prediction]]: label 하나가 아니라 가능한 출력의 분포를 예측합니다.
- [[concepts/machine-learning/decision-rule|Decision rule]]: probability, score, uncertainty를 실제 action으로 바꿉니다.
- [[concepts/machine-learning/classification|Classification]]: discrete label을 예측합니다.
- [[concepts/machine-learning/regression|Regression]]: continuous value를 예측합니다.
- [[concepts/machine-learning/ranking|Ranking]]: 후보들을 순서화합니다.
- [[concepts/machine-learning/clustering|Clustering]]: label 없이 비슷한 sample을 묶습니다.
- [[concepts/machine-learning/density-estimation|Density estimation]]: data distribution 자체를 모델링합니다.
- [[concepts/machine-learning/representation-learning|Representation learning]]: downstream task에 쓸 embedding을 학습합니다.

## 모델 계열

- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/machine-learning/linear-model|Linear model]]
- [[concepts/machine-learning/tree-based-model|Tree-based model]]
- [[concepts/machine-learning/kernel-method|Kernel method]]
- [[concepts/machine-learning/ensemble-method|Ensemble method]]
- [[concepts/architectures/mlp|MLP]]

## 학습의 기본 요소

- Objective: 무엇을 줄이거나 키울 것인가
- [[concepts/machine-learning/empirical-risk-minimization|Empirical risk minimization]]: observed data에서 평균 손실을 줄이는 기본 원리
- [[concepts/machine-learning/loss-function|Loss function]]: prediction error를 어떻게 수치화할 것인가
- [[concepts/machine-learning/cross-entropy-loss|Cross-entropy loss]], [[concepts/machine-learning/mean-squared-error|mean squared error]], [[concepts/machine-learning/negative-log-likelihood|negative log-likelihood]]: task와 probability assumption에 따라 loss가 어떻게 달라지는가
- [[concepts/machine-learning/training-loop|Training loop]]: forward, loss, backward, update를 어떻게 반복할 것인가
- [[concepts/machine-learning/optimization|Optimization]]: parameter를 어떻게 업데이트할 것인가
- [[concepts/machine-learning/stochastic-gradient|Stochastic gradient]]: mini-batch로 full gradient를 어떻게 추정할 것인가
- [[concepts/machine-learning/backpropagation|Backpropagation]]: loss에서 parameter gradient를 어떻게 계산할 것인가
- [[concepts/machine-learning/optimizer|Optimizer]]: gradient를 실제 update로 어떻게 바꿀 것인가
- [[concepts/machine-learning/adam|Adam]]과 [[concepts/machine-learning/adamw|AdamW]]: adaptive moment와 decoupled weight decay를 어떻게 해석할 것인가
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]: update 크기를 시간에 따라 어떻게 조절할 것인가
- [[concepts/machine-learning/training-stability|Training stability]]: loss, gradient norm, learning rate, batch size, checkpoint resume가 안정적으로 맞물리는가
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]: micro-step, optimizer step, consumed samples/tokens를 어떻게 구분할 것인가
- [[concepts/machine-learning/weight-decay|Weight decay]]: parameter 크기를 어떻게 제한할 것인가
- [[concepts/machine-learning/gradient-clipping|Gradient clipping]]: 불안정한 gradient를 어떻게 제한할 것인가
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]: memory limit 안에서 effective batch size를 어떻게 만들 것인가
- [[concepts/machine-learning/batch-size|Batch size]]: gradient estimate와 memory/throughput을 어떻게 trade-off할 것인가
- [[concepts/machine-learning/generalization|Generalization]]: train data 밖에서도 성능이 유지된다는 claim을 어떻게 정의할 것인가
- [[concepts/machine-learning/overfitting-underfitting|Overfitting and underfitting]]: train/validation/test curve를 어떻게 해석할 것인가
- [[concepts/machine-learning/model-selection|Model selection]]: 여러 후보 중 최종 모델을 어떤 근거로 고를 것인가
- [[concepts/machine-learning/hyperparameter-tuning|Hyperparameter tuning]]: learning rate, weight decay, model size 같은 선택을 어떤 budget 안에서 탐색할 것인가
- [[concepts/machine-learning/early-stopping|Early stopping]]: validation metric으로 checkpoint를 고를 때 어떤 leakage를 피해야 하는가
- [[concepts/machine-learning/learning-curve|Learning curve]]: train/validation curve로 optimization, underfit, overfit을 어떻게 구분할 것인가
- [[concepts/machine-learning/validation-curve|Validation curve]]: hyperparameter 변화에 따른 train/validation 성능을 어떻게 해석할 것인가
- [[concepts/machine-learning/regularization|Regularization]]: overfitting을 어떻게 줄일 것인가
- [[concepts/evaluation/train-validation-test-split|Validation split]]: model selection을 어떤 split에서 할 것인가
- [[concepts/systems/experiment-lifecycle|Experiment lifecycle]]: 질문, 가설, 설계, run, artifact, 분석, claim을 어떻게 연결할 것인가
- [[concepts/systems/run-artifact|Run artifact]]: config, logs, metrics, predictions, checkpoint, environment를 어떤 수준으로 남길 것인가

일반화 관점에서는 train loss와 test loss를 구분해야 합니다.

$$
\mathrm{gap}
= \mathcal{L}_{\mathrm{test}}(f_\theta)
- \mathcal{L}_{\mathrm{train}}(f_\theta)
$$

gap이 크면 모델이 training data의 우연한 패턴에 맞춰졌을 가능성을 의심합니다.

## Representation과 Feature

Classical ML에서는 feature design이 중심이고, deep learning에서는 feature extractor 자체를 학습합니다. 이 차이를 이해하면 [[concepts/learning/self-supervised-learning|self-supervised learning]]이나 [[concepts/learning/transfer-learning|transfer learning]]이 왜 중요한지 더 분명해집니다.

- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/transfer-learning|Transfer learning]]

## 평가 질문

- train/test split이 deployment 상황을 반영하는가?
- example unit, split unit, label semantics가 명확한가?
- metric이 실제 목표와 맞는가?
- [[concepts/evaluation/baseline|baseline]]과 비교했을 때 개선이 의미 있는가?
- [[concepts/evaluation/ablation-study|ablation study]]가 설계 선택을 지지하는가?
- overfitting, leakage, distribution shift를 확인했는가?
- generalization claim이 IID인지, OOD인지, deployment claim인지 명확한가?

## Related

- [[ai/learning-methods|Learning methods]]
- [[ai/evaluation|Evaluation]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/machine-learning/probabilistic-prediction|Probabilistic prediction]]
- [[concepts/machine-learning/decision-rule|Decision rule]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/machine-learning/generalization|Generalization]]
- [[concepts/machine-learning/overfitting-underfitting|Overfitting and underfitting]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/machine-learning/hyperparameter-tuning|Hyperparameter tuning]]
- [[concepts/machine-learning/early-stopping|Early stopping]]
- [[concepts/machine-learning/learning-curve|Learning curve]]
- [[concepts/machine-learning/validation-curve|Validation curve]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/systems/experiment-lifecycle|Experiment lifecycle]]
- [[concepts/systems/run-artifact|Run artifact]]
