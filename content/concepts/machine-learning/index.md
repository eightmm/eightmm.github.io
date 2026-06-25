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

## Model Families

- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/machine-learning/density-estimation|Density estimation]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/machine-learning/linear-model|Linear model]]
- [[concepts/machine-learning/tree-based-model|Tree-based model]]
- [[concepts/machine-learning/kernel-method|Kernel method]]
- [[concepts/machine-learning/ensemble-method|Ensemble method]]
- [[concepts/architectures/mlp|MLP]]

## Methods

- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/clustering|Clustering]]
- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]

## Related

- [[concepts/tasks/index|Tasks]]
- [[concepts/learning/index|Learning methods]]
- [[concepts/architectures/index|Architectures]]
- [[concepts/evaluation/index|Evaluation]]
