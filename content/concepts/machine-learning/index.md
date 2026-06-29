---
title: Machine Learning
tags:
  - machine-learning
---

# Machine Learning

Machine learning은 data로부터 prediction이나 decision을 개선하는 algorithm을 다룹니다. 이 wiki에서는 architecture, learning objective, generative model, evaluation의 기반층입니다.

표준 supervised learning 설정은 empirical risk minimization으로 쓸 수 있습니다.

$$
\hat{\theta}
= \arg\min_\theta
\frac{1}{n}\sum_{i=1}^{n}
\mathcal{L}(f_\theta(x_i), y_i)
$$

여기서 $f_\theta$는 model, $\mathcal{L}$은 loss, $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^n$는 training set입니다.

## 핵심 구성

- Data: example, label, feature, split, sampling process.
- Model: input을 prediction 또는 representation으로 바꾸는 function class.
- Objective: training 중 최적화하는 quantity.
- Optimization: model parameter를 업데이트하는 절차.
- Evaluation: generalization을 추정하는 절차.
- Generalization: claim의 대상이 되는 held-out 또는 deployment behavior.

## 이동 지도

| 질문 | 시작점 | 이어서 확인할 것 |
| --- | --- | --- |
| what is one training example? | [Data](/concepts/data), [Data preprocessing](/concepts/machine-learning/data-preprocessing) | [Example unit](/concepts/data/example-unit), [Split unit](/concepts/data/split-unit) |
| what does the model output? | [Probabilistic prediction](/concepts/machine-learning/probabilistic-prediction), [Decision rule](/concepts/machine-learning/decision-rule) | [Task output space](/concepts/tasks/task-output-space) |
| is it classification, regression, or ranking? | [Classification](/concepts/machine-learning/classification), [Regression](/concepts/machine-learning/regression), [Ranking](/concepts/machine-learning/ranking) | metric and threshold policy |
| what objective is optimized? | [Empirical risk minimization](/concepts/machine-learning/empirical-risk-minimization), [Loss function](/concepts/machine-learning/loss-function) | [Objective-metric alignment](/concepts/machine-learning/objective-metric-alignment) |
| is the loss probabilistic? | [Negative log-likelihood](/concepts/machine-learning/negative-log-likelihood), [Cross-entropy loss](/concepts/machine-learning/cross-entropy-loss), [Mean squared error](/concepts/machine-learning/mean-squared-error) | likelihood family and reduction boundary |
| how are parameters updated? | [Optimization](/concepts/machine-learning/optimization), [Gradient descent](/concepts/machine-learning/gradient-descent), [Backpropagation](/concepts/machine-learning/backpropagation) | learning rate, batch size, gradient scale |
| how is generalization estimated? | [Generalization](/concepts/machine-learning/generalization), [Model selection](/concepts/machine-learning/model-selection) | [Train/validation/test split](/concepts/evaluation/train-validation-test-split), [Leakage](/concepts/evaluation/leakage) |
| why did training fail? | [Training stability](/concepts/machine-learning/training-stability), [Loss landscape](/concepts/machine-learning/loss-landscape) | gradients, optimizer state, data scale |

## 기반

| 그룹 | 노트 |
| --- | --- |
| Probability and estimators | [Random variable](/concepts/math/random-variable), [Statistical estimator](/concepts/math/statistical-estimator), [Maximum likelihood](/concepts/math/maximum-likelihood) |
| Data and features | [Data preprocessing](/concepts/machine-learning/data-preprocessing), [Feature engineering](/concepts/machine-learning/feature-engineering), [Dataset shift](/concepts/data/dataset-shift) |
| Prediction tasks | [Classification](/concepts/machine-learning/classification), [Regression](/concepts/machine-learning/regression), [Ranking](/concepts/machine-learning/ranking), [Density estimation](/concepts/machine-learning/density-estimation) |
| Classical models | [Linear model](/concepts/machine-learning/linear-model), [Tree-based model](/concepts/machine-learning/tree-based-model), [Kernel method](/concepts/machine-learning/kernel-method), [Ensemble method](/concepts/machine-learning/ensemble-method) |
| Representations | [Representation learning](/concepts/machine-learning/representation-learning), [Dimensionality reduction](/concepts/machine-learning/dimensionality-reduction), [MLP](/concepts/architectures/mlp) |

## Training과 Selection

| 그룹 | 노트 |
| --- | --- |
| Objective | [Empirical risk minimization](/concepts/machine-learning/empirical-risk-minimization), [Loss function](/concepts/machine-learning/loss-function), [Objective-metric alignment](/concepts/machine-learning/objective-metric-alignment) |
| Losses | [Cross-entropy loss](/concepts/machine-learning/cross-entropy-loss), [Mean squared error](/concepts/machine-learning/mean-squared-error), [Negative log-likelihood](/concepts/machine-learning/negative-log-likelihood) |
| Optimization | [Optimization](/concepts/machine-learning/optimization), [Stochastic gradient](/concepts/machine-learning/stochastic-gradient), [Gradient descent](/concepts/machine-learning/gradient-descent), [Second-order optimization](/concepts/machine-learning/second-order-optimization) |
| Gradients | [Backpropagation](/concepts/machine-learning/backpropagation), [Automatic differentiation](/concepts/machine-learning/automatic-differentiation), [Gradient checking](/concepts/machine-learning/gradient-checking) |
| Optimizers | [Optimizer](/concepts/machine-learning/optimizer), [Adam](/concepts/machine-learning/adam), [AdamW](/concepts/machine-learning/adamw), [Learning rate schedule](/concepts/machine-learning/learning-rate-schedule) |
| Stability | [Training loop](/concepts/machine-learning/training-loop), [Training stability](/concepts/machine-learning/training-stability), [Training step accounting](/concepts/machine-learning/training-step-accounting), [Model state contract](/concepts/machine-learning/model-state-contract) |
| Regularization | [Regularization](/concepts/machine-learning/regularization), [Weight decay](/concepts/machine-learning/weight-decay), [Gradient clipping](/concepts/machine-learning/gradient-clipping), [Gradient accumulation](/concepts/machine-learning/gradient-accumulation), [Batch size](/concepts/machine-learning/batch-size) |
| Model choice | [Model selection](/concepts/machine-learning/model-selection), [Hyperparameter tuning](/concepts/machine-learning/hyperparameter-tuning), [Early stopping](/concepts/machine-learning/early-stopping), [Learning curve](/concepts/machine-learning/learning-curve), [Validation curve](/concepts/machine-learning/validation-curve) |
| Failure modes | [Overfitting and underfitting](/concepts/machine-learning/overfitting-underfitting), [Bias-variance tradeoff](/concepts/math/bias-variance-tradeoff), [Leakage](/concepts/evaluation/leakage) |

## Related

- [[concepts/math/index|Math foundations]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/data/index|Data]]
- [[concepts/learning/index|Learning methods]]
- [[concepts/architectures/index|Architectures]]
- [[concepts/evaluation/index|Evaluation]]
