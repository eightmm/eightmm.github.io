---
title: Covariance and Correlation
tags:
  - math
  - probability
  - statistics
---

# Covariance and Correlation

Covariance measures how two random variables vary together. Correlation normalizes covariance so the scale is easier to compare.

For random variables $X$ and $Y$:

$$
\operatorname{Cov}(X,Y)
= \mathbb{E}\left[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])\right]
$$

The correlation coefficient is:

$$
\rho_{X,Y}
=
\frac{\operatorname{Cov}(X,Y)}
{\sqrt{\operatorname{Var}(X)}\sqrt{\operatorname{Var}(Y)}}
$$

where:

$$
\operatorname{Var}(X)
= \operatorname{Cov}(X,X)
= \mathbb{E}\left[(X-\mathbb{E}[X])^2\right]
$$

The empirical covariance from paired samples $(x_i,y_i)$ is:

$$
\widehat{\operatorname{Cov}}(X,Y)
=
\frac{1}{n-1}
\sum_{i=1}^{n}
(x_i-\bar{x})(y_i-\bar{y})
$$

where:

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i,
\quad
\bar{y}=\frac{1}{n}\sum_{i=1}^{n}y_i
$$

## Vector Case

For a random vector $x \in \mathbb{R}^d$ with mean $\mu=\mathbb{E}[x]$, the covariance matrix is:

$$
\Sigma
= \mathbb{E}\left[(x-\mu)(x-\mu)^\top\right]
$$

The diagonal entries are variances. Off-diagonal entries describe pairwise covariance.

The sample covariance matrix is:

$$
\hat{\Sigma}
=
\frac{1}{n-1}
\sum_{i=1}^{n}
(x_i-\bar{x})(x_i-\bar{x})^\top
$$

For centered data matrix $X_c\in\mathbb{R}^{n\times d}$:

$$
\hat{\Sigma}
=
\frac{1}{n-1}X_c^\top X_c
$$

This connects covariance directly to [[concepts/math/eigenvalue-eigenvector|eigenvalue decomposition]] and [[concepts/math/singular-value-decomposition|singular value decomposition]].

## Correlation Matrix

If $D$ is the diagonal matrix of feature standard deviations, the correlation matrix is:

$$
R
=
D^{-1}\Sigma D^{-1}
$$

Correlation is useful when feature scales are arbitrary, but it can hide absolute variance differences that matter for optimization or measurement noise.

## Representation Analysis

Embedding covariance is often used to detect collapse or redundancy. If all embeddings become similar, the covariance spectrum becomes low-rank:

$$
\lambda_1 \gg \lambda_2,\dots,\lambda_d
$$

where $\lambda_j$ are eigenvalues of $\hat{\Sigma}$. This can indicate that a representation has lost useful degrees of freedom.

## Why It Matters

- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]] often studies covariance structure.
- Feature correlation can make linear models unstable.
- Evaluation results can be misleading when samples are not independent.
- Representation analysis often compares embedding covariance or collapse.
- Covariance assumptions appear in Gaussian models, uncertainty estimates, and whitening.

## Watch For

- Correlation is not causation.
- Zero covariance does not always mean independence.
- Strong correlation can be an artifact of leakage, duplicates, or confounders.
- Covariance depends on scale; correlation removes scale but not bias.
- Pairwise correlation can miss nonlinear dependence.
- Dataset shift can change covariance even when means look stable.

## Related

- [[concepts/math/random-variable|Random variable]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/math/singular-value-decomposition|Singular value decomposition]]
- [[concepts/math/normal-distribution|Normal distribution]]
- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]
- [[concepts/machine-learning/linear-model|Linear model]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/evaluation/leakage|Leakage]]
