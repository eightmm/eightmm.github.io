---
title: Linear Model
tags:
  - machine-learning
---

# Linear Model

A linear model predicts from a weighted sum of input features. It is simple, interpretable, and often a strong baseline.

The basic form is:

$$
\hat{y} = w^\top x + b
$$

where $x$ is a feature vector, $w$ is a weight vector, and $b$ is a bias term.

## Common Forms

- Linear regression predicts a continuous value.
- Logistic regression predicts class probabilities.
- Linear classifiers separate classes with a hyperplane.

For binary logistic regression:

$$
p(y=1\mid x)
= \sigma(w^\top x + b)
= \frac{1}{1+\exp(-(w^\top x+b))}
$$

## Why It Matters

- It makes the role of features explicit.
- It is fast to train and easy to debug.
- It provides a baseline before using larger architectures.

## Watch For

- Linear assumptions may be too restrictive.
- Feature scaling can matter.
- Correlated features can make coefficients hard to interpret.

## Related

- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/kernel-method|Kernel method]]
- [[concepts/architectures/mlp|MLP]]
