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

For linear regression with squared error, the empirical objective is:

$$
\hat{w},\hat{b}
=
\arg\min_{w,b}
\frac{1}{n}\sum_{i=1}^{n}
(w^\top x_i+b-y_i)^2
$$

The prediction changes by $w_j$ when feature $x_j$ increases by one unit, assuming other features are fixed. This makes the model easy to inspect, but only when feature scaling and feature correlations are understood.

For binary logistic regression:

$$
p(y=1\mid x)
= \sigma(w^\top x + b)
= \frac{1}{1+\exp(-(w^\top x+b))}
$$

The decision boundary at threshold $\tau=0.5$ is:

$$
w^\top x+b=0
$$

For multiclass logistic regression:

$$
z = Wx+b,
\qquad
p(y=k\mid x)=
\frac{\exp(z_k)}{\sum_{\ell=1}^{K}\exp(z_\ell)}
$$

## Why It Matters

- It makes the role of features explicit.
- It is fast to train and easy to debug.
- It provides a baseline before using larger architectures.
- It separates feature quality from model capacity.

## Checks

- Are features standardized when scale-sensitive optimization or regularization is used?
- Are coefficients interpreted only after checking collinearity and preprocessing?
- Is the decision threshold selected on validation data rather than test data?
- Does a nonlinear model outperform this baseline for the right reason, not leakage?

## Watch For

- Linear assumptions may be too restrictive.
- Feature scaling can matter.
- Correlated features can make coefficients hard to interpret.

## Related

- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/machine-learning/kernel-method|Kernel method]]
- [[concepts/architectures/mlp|MLP]]
