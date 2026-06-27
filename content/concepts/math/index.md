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

Use these pages to identify the object, distribution, parameter, and estimator behind a formula. Proof-heavy material can be added later only when it clarifies an AI, Computational Biology, or evaluation note.

## Core Notes

| Area | Use For | Start |
| --- | --- | --- |
| Linear algebra | vectors, matrices, projections, embeddings, rank, similarity | [Linear algebra](/concepts/math/linear-algebra), [Tensor shape notation](/concepts/math/tensor-shape-notation), [Vector norm and similarity](/concepts/math/vector-norm-similarity) |
| Spectral structure | representation analysis, PCA/SVD, low-rank approximations, stability | [Eigenvalue and eigenvector](/concepts/math/eigenvalue-eigenvector), [Singular value decomposition](/concepts/math/singular-value-decomposition) |
| Calculus | gradients, Jacobians, Hessians, backpropagation notation | [Calculus](/concepts/math/calculus), [Matrix calculus](/concepts/math/matrix-calculus), [Jacobian and Hessian](/concepts/math/jacobian-hessian) |
| Geometry and symmetry | coordinate sets, rigid motions, equivariance, invariant targets | [Geometry](/concepts/math/geometry), [Symmetry group](/concepts/math/symmetry-group) |
| Probability | random variables, distributions, expectations, normal assumptions | [Random variable](/concepts/math/random-variable), [Probability distribution](/concepts/math/probability-distribution), [Expectation](/concepts/math/expectation), [Normal distribution](/concepts/math/normal-distribution) |
| Estimation and statistics | finite-sample estimates, uncertainty, hypothesis checks, bias/variance | [Statistical estimator](/concepts/math/statistical-estimator), [Monte Carlo estimation](/concepts/math/monte-carlo-estimation), [Central limit theorem](/concepts/math/central-limit-theorem), [Hypothesis testing](/concepts/math/hypothesis-testing), [Bias-variance tradeoff](/concepts/math/bias-variance-tradeoff) |
| Dependence | covariance, correlation, representation diagnostics, label relationships | [Covariance and correlation](/concepts/math/covariance-correlation) |
| Likelihood and information | MLE, entropy, KL, cross-entropy, Bayesian updates | [Maximum likelihood](/concepts/math/maximum-likelihood), [Entropy and KL divergence](/concepts/math/entropy-kl), [Bayes rule](/concepts/math/bayes-rule), [Bayesian inference](/concepts/math/bayesian-inference) |

## Checks

- What object is the formula about: scalar, vector, matrix, graph, distribution, or function?
- What does each tensor axis mean, and which axis is reduced, mixed, normalized, or broadcast?
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
- State tensor shape and axis semantics when the operation depends on batch, token, graph, coordinate, head, or candidate axes.
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
