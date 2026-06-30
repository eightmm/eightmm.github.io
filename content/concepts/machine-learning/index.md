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

## Machine Learning Contract

Machine learning note는 데이터에서 claim까지 이어지는 계약을 분리해서 써야 합니다.

$$
\mathcal{M}
=
(\mathcal{D},\ \phi,\ f_\theta,\ \mathcal{Y},\ \mathcal{L},\ O,\ E)
$$

| Part | Meaning | Typical question |
| --- | --- | --- |
| $\mathcal{D}$ | dataset and sampling process | example unit, label semantics, split unit이 무엇인가? |
| $\phi$ | representation or preprocessing | raw input이 feature, token, graph, coordinate, embedding 중 무엇이 되는가? |
| $f_\theta$ | model family | 어떤 function class와 inductive bias를 쓰는가? |
| $\mathcal{Y}$ | output space | class, scalar, ranking, distribution, embedding, action 중 무엇인가? |
| $\mathcal{L}$ | training objective | parameter update가 실제로 줄이는 quantity는 무엇인가? |
| $O$ | optimizer and training state | gradient, batch, learning rate, checkpoint state가 어떻게 관리되는가? |
| $E$ | evaluation protocol | metric, split, baseline, uncertainty가 claim을 지지하는가? |

이 계약이 명확하지 않으면 모델 이름이 같아도 서로 다른 문제를 푸는 것입니다.

$$
\text{model comparison}
\Rightarrow
\text{same data, task, objective, and evaluation boundary}
$$

## 핵심 구성

- Data: example, label, feature, split, sampling process.
- Model: input을 prediction 또는 representation으로 바꾸는 function class.
- Objective: training 중 최적화하는 quantity.
- Optimization: model parameter를 업데이트하는 절차.
- Evaluation: generalization을 추정하는 절차.
- Generalization: claim의 대상이 되는 held-out 또는 deployment behavior.

## Prediction, Objective, Decision

Prediction, training objective, decision rule은 서로 다른 층입니다.

$$
x
\xrightarrow{f_\theta}
\hat{p}_\theta(y\mid x)
\xrightarrow{\mathcal{L}}
\text{parameter update}
$$

Deployment에서는 prediction을 action으로 바꿉니다.

$$
\hat{a}
=
\delta(\hat{p}_\theta(y\mid x),\ \tau,\ C)
$$

여기서 $\delta$는 decision rule, $\tau$는 threshold, $C$는 cost 또는 constraint입니다.

| Layer | Example | Main risk |
| --- | --- | --- |
| Prediction | class probability, score, regression value, embedding | output semantics가 불명확함 |
| Objective | cross-entropy, MSE, NLL, ranking loss | metric 또는 utility와 불일치 |
| Optimization | SGD, Adam, AdamW, schedule, clipping | instability 또는 hidden state 누락 |
| Decision | threshold, top-k, reject option, calibration policy | probability를 action처럼 오해 |
| Evaluation | held-out metric, calibration, uncertainty, failure slice | selection leakage 또는 weak split |

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

## Boundary With Other Sections

| If the note is about | Put it in |
| --- | --- |
| architecture internals, layers, attention, SSM, GNN | [[concepts/architectures/index|Architectures]] |
| supervision style, SSL, transfer, active learning, curriculum | [[concepts/learning/index|Learning methods]] |
| sampling from a modeled distribution | [[concepts/generative-models/index|Generative models]] |
| task definition and output semantics | [[concepts/tasks/index|Tasks]] |
| dataset construction, split, label provenance | [[concepts/data/index|Data]] |
| metric, baseline, uncertainty, leakage, OOD claim | [[concepts/evaluation/index|Evaluation]] |
| run artifact, checkpoint, environment, serving | [[concepts/systems/index|AI systems]] |
| domain-specific object like molecule, protein, pocket | [[molecular-modeling/index|Computational Biology]] |

Machine learning is the shared grammar across these sections. Keep it focused on prediction, objective, optimization, model selection, and generalization.

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
