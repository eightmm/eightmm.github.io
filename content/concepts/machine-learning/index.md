---
title: Machine Learning
tags:
  - machine-learning
---

# Machine Learning

Machine learning studies algorithms that improve predictions or decisions from data. In this wiki, it is the base layer under architectures, learning objectives, generative models, and evaluation.

The standard supervised learning setup is empirical risk minimization:

$$
\hat{\theta}
= \arg\min_\theta
\frac{1}{n}\sum_{i=1}^{n}
\mathcal{L}(f_\theta(x_i), y_i)
$$

Here $f_\theta$ is the model, $\mathcal{L}$ is the loss, and $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^n$ is the training set.

## Core Pieces

- Data: examples, labels, features, splits, and sampling process.
- Model: a function class that maps inputs to predictions or representations.
- Objective: the quantity optimized during training.
- Optimization: the procedure used to update model parameters.
- Evaluation: the procedure used to estimate generalization.
- Generalization: the held-out or deployment behavior being claimed.

## Route Map

| Question | Start | Then Check |
| --- | --- | --- |
| what is one training example? | [Data](/concepts/data), [Data preprocessing](/concepts/machine-learning/data-preprocessing) | [Example unit](/concepts/data/example-unit), [Split unit](/concepts/data/split-unit) |
| what does the model output? | [Probabilistic prediction](/concepts/machine-learning/probabilistic-prediction), [Decision rule](/concepts/machine-learning/decision-rule) | [Task output space](/concepts/tasks/task-output-space) |
| is it classification, regression, or ranking? | [Classification](/concepts/machine-learning/classification), [Regression](/concepts/machine-learning/regression), [Ranking](/concepts/machine-learning/ranking) | metric and threshold policy |
| what objective is optimized? | [Empirical risk minimization](/concepts/machine-learning/empirical-risk-minimization), [Loss function](/concepts/machine-learning/loss-function) | [Objective-metric alignment](/concepts/machine-learning/objective-metric-alignment) |
| is the loss probabilistic? | [Negative log-likelihood](/concepts/machine-learning/negative-log-likelihood), [Cross-entropy loss](/concepts/machine-learning/cross-entropy-loss), [Mean squared error](/concepts/machine-learning/mean-squared-error) | likelihood family and reduction boundary |
| how are parameters updated? | [Optimization](/concepts/machine-learning/optimization), [Gradient descent](/concepts/machine-learning/gradient-descent), [Backpropagation](/concepts/machine-learning/backpropagation) | learning rate, batch size, gradient scale |
| how is generalization estimated? | [Generalization](/concepts/machine-learning/generalization), [Model selection](/concepts/machine-learning/model-selection) | [Train/validation/test split](/concepts/evaluation/train-validation-test-split), [Leakage](/concepts/evaluation/leakage) |
| why did training fail? | [Training stability](/concepts/machine-learning/training-stability), [Loss landscape](/concepts/machine-learning/loss-landscape) | gradients, optimizer state, data scale |

## Foundations

| Group | Notes |
| --- | --- |
| Probability and estimators | [Random variable](/concepts/math/random-variable), [Statistical estimator](/concepts/math/statistical-estimator), [Maximum likelihood](/concepts/math/maximum-likelihood) |
| Data and features | [Data preprocessing](/concepts/machine-learning/data-preprocessing), [Feature engineering](/concepts/machine-learning/feature-engineering), [Dataset shift](/concepts/data/dataset-shift) |
| Prediction tasks | [Classification](/concepts/machine-learning/classification), [Regression](/concepts/machine-learning/regression), [Ranking](/concepts/machine-learning/ranking), [Density estimation](/concepts/machine-learning/density-estimation) |
| Classical models | [Linear model](/concepts/machine-learning/linear-model), [Tree-based model](/concepts/machine-learning/tree-based-model), [Kernel method](/concepts/machine-learning/kernel-method), [Ensemble method](/concepts/machine-learning/ensemble-method) |
| Representations | [Representation learning](/concepts/machine-learning/representation-learning), [Dimensionality reduction](/concepts/machine-learning/dimensionality-reduction), [MLP](/concepts/architectures/mlp) |

## Training and Selection

| Group | Notes |
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
