---
title: Math Foundations
tags:
  - math
  - foundations
  - machine-learning
---

# Math Foundations

Math foundation notes collect the minimum reusable definitions needed to read AI papers and implementation notes. The goal is not to build a full math textbook, but to make formulas in architecture, learning, generation, and evaluation notes easier to parse.

Most machine learning methods combine:

$$
\text{representation} + \text{probability model} + \text{objective} + \text{optimization}
$$

Use these pages to identify the object, distribution, parameter, and estimator behind a formula. Proof-heavy material can be added later only when it clarifies an AI, Bio-AI, or evaluation note.

## Core Notes

- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/math/vector-norm-similarity|Vector norm and similarity]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/math/singular-value-decomposition|Singular value decomposition]]
- [[concepts/math/calculus|Calculus]]
- [[concepts/math/matrix-calculus|Matrix calculus]]
- [[concepts/math/jacobian-hessian|Jacobian and Hessian]]
- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/math/random-variable|Random variable]]
- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/normal-distribution|Normal distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/math/central-limit-theorem|Central limit theorem]]
- [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/math/bayes-rule|Bayes rule]]
- [[concepts/math/statistical-estimator|Statistical estimator]]
- [[concepts/math/hypothesis-testing|Hypothesis testing]]
- [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]]

## Checks

- What object is the formula about: scalar, vector, matrix, graph, distribution, or function?
- Is a derivative with respect to inputs, parameters, coordinates, or time?
- Is the quantity optimized, estimated, sampled, normalized, or measured?
- Is a probability statement conditional or marginal?
- Is a loss a negative log-likelihood, distance, ranking loss, or surrogate?
- Are expectations taken under data, model, sampling, or deployment distributions?
- If the object has coordinates, what transformations should preserve meaning?

## Formula Reading Template

When adding or expanding a concept page, prefer formulas that define the quantity directly.

- Define symbols before using them.
- Separate population quantities from finite-sample estimates.
- State the distribution behind each expectation.
- State whether a matrix/vector shape matters.
- Name the variable being optimized or differentiated.
- Link the formula to the modeling decision it supports.

Example:

$$
\hat{R}(f)
=
\frac{1}{m}\sum_{j=1}^{m}
\mathcal{L}(f(x_j), y_j)
$$

This is an empirical estimate of test risk. It is not the same object as the population risk

$$
R(f)
=
\mathbb{E}_{(x,y)\sim p_{\mathrm{test}}}
[\mathcal{L}(f(x), y)].
$$

## Related

- [[concepts/machine-learning/index|Machine learning]]
- [[concepts/generative-models/index|Generative models]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/architectures/index|Architectures]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
