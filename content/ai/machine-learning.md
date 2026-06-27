---
title: Machine Learning
tags:
  - ai
  - machine-learning
---

# Machine Learning

Machine learning은 AI를 이해하기 위한 기본 문법입니다. 복잡한 deep learning model도 결국 data, feature, objective, optimization, evaluation의 조합으로 봐야 합니다.

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

## Problem to Loss Map

| Problem | Output | Common Loss | Evaluation |
| --- | --- | --- | --- |
| Classification | class probability $p_\theta(y\mid x)$ | cross-entropy / NLL | accuracy, F1, AUROC, PR-AUC, calibration |
| Regression | scalar or vector $\hat{y}$ | MSE, MAE, Gaussian NLL | RMSE, MAE, $R^2$, rank correlation |
| Ranking | ordered candidate list | pairwise/listwise ranking loss | NDCG, MAP, enrichment, top-k success |
| Density estimation | probability model $p_\theta(x)$ | negative log-likelihood | held-out NLL, sampling quality |
| Representation learning | embedding $z=f_\theta(x)$ | contrastive, masked, predictive, reconstruction loss | linear probe, retrieval, transfer task |
| Clustering | cluster assignment or latent grouping | reconstruction, mixture likelihood, contrastive proxy | purity, ARI/NMI, downstream utility |

For probabilistic prediction, the model should expose uncertainty before a hard decision:

$$
p_\theta(y\mid x)
\rightarrow
\hat{y}=\arg\max_y p_\theta(y\mid x)
$$

The decision rule $\hat{y}$ is not the same object as the probability distribution $p_\theta(y\mid x)$.

## 기본 구성

Machine learning note는 아래 다섯 층을 분리해서 읽습니다.

| Layer | Question | Start |
| --- | --- | --- |
| Data contract | example, label, split, preprocessing이 무엇인가? | [Dataset checklist](/concepts/data/dataset-construction-checklist), [Example unit](/concepts/data/example-unit), [Split unit](/concepts/data/split-unit), [Label semantics](/concepts/data/label-semantics) |
| Representation | raw input이 feature, token, graph, coordinate, embedding으로 어떻게 바뀌는가? | [Data preprocessing](/concepts/machine-learning/data-preprocessing), [Feature engineering](/concepts/machine-learning/feature-engineering), [Representation learning](/concepts/machine-learning/representation-learning) |
| Model family | 어떤 함수공간과 inductive bias를 쓰는가? | [Linear model](/concepts/machine-learning/linear-model), [Tree-based model](/concepts/machine-learning/tree-based-model), [Kernel method](/concepts/machine-learning/kernel-method), [Ensemble method](/concepts/machine-learning/ensemble-method), [MLP](/concepts/architectures/mlp) |
| Objective | 무엇을 줄이거나 키우는가? | [ERM](/concepts/machine-learning/empirical-risk-minimization), [Loss function](/concepts/machine-learning/loss-function), [Objective-metric alignment](/concepts/machine-learning/objective-metric-alignment) |
| Evidence | 학습된 모델이 어떤 범위에서 작동한다고 말할 수 있는가? | [Generalization](/concepts/machine-learning/generalization), [Model selection](/concepts/machine-learning/model-selection), [Evaluation](/ai/evaluation) |

## Loss and Objective

Objective는 모델이 실제로 학습하는 신호입니다. 같은 architecture라도 objective가 달라지면 representation과 failure mode가 달라집니다.

| Objective Family | Use For | Notes |
| --- | --- | --- |
| Empirical risk | supervised prediction under observed labels | [ERM](/concepts/machine-learning/empirical-risk-minimization), [Loss function](/concepts/machine-learning/loss-function) |
| Classification likelihood | class probability prediction | [Cross-entropy loss](/concepts/machine-learning/cross-entropy-loss), [Negative log-likelihood](/concepts/machine-learning/negative-log-likelihood) |
| Regression error | scalar or vector prediction | [Mean squared error](/concepts/machine-learning/mean-squared-error), [Regression](/concepts/machine-learning/regression) |
| Ranking objective | ordered candidate list | [Ranking](/concepts/machine-learning/ranking), [Ranking metrics](/concepts/evaluation/ranking-metrics) |
| Representation objective | embedding useful for transfer, retrieval, or probing | [Representation learning](/concepts/machine-learning/representation-learning), [Self-supervised learning](/concepts/learning/self-supervised-learning) |
| Alignment check | reported metric differs from training loss | [Objective-metric alignment](/concepts/machine-learning/objective-metric-alignment) |

## Optimization and Training

Training connects the objective to parameter updates. The minimum loop is:

$$
x_B \rightarrow f_\theta(x_B) \rightarrow \mathcal{L}_B
\rightarrow \nabla_\theta \mathcal{L}_B
\rightarrow \theta_{t+1}
$$

| Topic | Ask | Start |
| --- | --- | --- |
| Training loop | forward, loss, backward, update가 어떤 순서로 도는가? | [Training loop](/concepts/machine-learning/training-loop) |
| Gradient estimate | full gradient가 아니라 어떤 mini-batch estimate를 쓰는가? | [Stochastic gradient](/concepts/machine-learning/stochastic-gradient), [Batch size](/concepts/machine-learning/batch-size) |
| Gradient computation | autograd와 backprop이 올바른가? | [Backpropagation](/concepts/machine-learning/backpropagation), [Automatic differentiation](/concepts/machine-learning/automatic-differentiation), [Gradient checking](/concepts/machine-learning/gradient-checking) |
| Update rule | gradient를 parameter update로 어떻게 바꾸는가? | [Optimization](/concepts/machine-learning/optimization), [Optimizer](/concepts/machine-learning/optimizer), [Adam](/concepts/machine-learning/adam), [AdamW](/concepts/machine-learning/adamw) |
| Stabilization | learning rate, clipping, weight decay, accumulation을 어떻게 쓰는가? | [Learning-rate schedule](/concepts/machine-learning/learning-rate-schedule), [Gradient clipping](/concepts/machine-learning/gradient-clipping), [Weight decay](/concepts/machine-learning/weight-decay), [Gradient accumulation](/concepts/machine-learning/gradient-accumulation) |
| Diagnostics | loss surface, curvature, instability를 어떻게 읽는가? | [Training stability](/concepts/machine-learning/training-stability), [Loss landscape](/concepts/machine-learning/loss-landscape), [Second-order optimization](/concepts/machine-learning/second-order-optimization) |
| State tracking | step, checkpoint, optimizer state, data state가 구분되는가? | [Training step accounting](/concepts/machine-learning/training-step-accounting), [Model state contract](/concepts/machine-learning/model-state-contract) |

## Generalization and Selection

성능 claim은 train loss가 아니라 선택 절차와 held-out evidence로 정해집니다.

| Question | Risk | Start |
| --- | --- | --- |
| train/validation/test가 분리되어 있는가? | validation decision이 test에 새어 들어갈 수 있음 | [Train/validation/test split](/concepts/evaluation/train-validation-test-split) |
| 어떤 후보 중 최종 모델을 골랐는가? | best checkpoint나 best seed를 test score처럼 읽을 수 있음 | [Model selection](/concepts/machine-learning/model-selection), [Hyperparameter tuning](/concepts/machine-learning/hyperparameter-tuning) |
| 언제 멈췄는가? | early stopping이 hidden test tuning이 될 수 있음 | [Early stopping](/concepts/machine-learning/early-stopping) |
| curve가 무엇을 말하는가? | optimization failure와 overfitting을 혼동할 수 있음 | [Learning curve](/concepts/machine-learning/learning-curve), [Validation curve](/concepts/machine-learning/validation-curve), [Overfitting and underfitting](/concepts/machine-learning/overfitting-underfitting) |
| 일반화 claim이 무엇인가? | IID, OOD, deployment claim을 섞을 수 있음 | [Generalization](/concepts/machine-learning/generalization), [OOD generalization](/concepts/evaluation/ood-generalization), [Leakage](/concepts/evaluation/leakage) |
| overfitting을 어떻게 줄였는가? | regularizer가 claim을 바꾸거나 baseline을 불공정하게 만들 수 있음 | [Regularization](/concepts/machine-learning/regularization) |

## Training State

Training is not only a loss value. A reproducible training note should identify:

| State | Meaning |
| --- | --- |
| Parameters $\theta_t$ | model weights at step $t$ |
| Optimizer state $m_t$ | momentum, variance estimate, or other update memory |
| Batch $B_t$ | examples used for one gradient estimate |
| Gradient $g_t$ | estimated direction from the current loss |
| Learning rate $\eta_t$ | update scale at step $t$ |
| Checkpoint | parameters, optimizer state, scheduler state, config, and step count |

The generic update is:

$$
\theta_{t+1}
=
\theta_t
-
\eta_t\,u(g_t, m_t)
$$

where $u(g_t,m_t)$ is the optimizer-specific update direction.

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
- training objective와 reported metric이 같은 claim을 지지하는가?
- overfitting, leakage, distribution shift를 확인했는가?
- generalization claim이 IID인지, OOD인지, deployment claim인지 명확한가?

## Related

- [[ai/learning-methods|Learning methods]]
- [[ai/evaluation|Evaluation]]
- [[concepts/machine-learning/objective-metric-alignment|Objective-metric alignment]]
- [[concepts/machine-learning/probabilistic-prediction|Probabilistic prediction]]
- [[concepts/machine-learning/decision-rule|Decision rule]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/machine-learning/generalization|Generalization]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/model-state-contract|Model state contract]]
- [[concepts/systems/experiment-lifecycle|Experiment lifecycle]]
- [[concepts/systems/run-artifact|Run artifact]]
