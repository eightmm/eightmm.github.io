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

## Vector Case

For a random vector $x \in \mathbb{R}^d$ with mean $\mu=\mathbb{E}[x]$, the covariance matrix is:

$$
\Sigma
= \mathbb{E}\left[(x-\mu)(x-\mu)^\top\right]
$$

The diagonal entries are variances. Off-diagonal entries describe pairwise covariance.

## Why It Matters

- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]] often studies covariance structure.
- Feature correlation can make linear models unstable.
- Evaluation results can be misleading when samples are not independent.
- Representation analysis often compares embedding covariance or collapse.

## Watch For

- Correlation is not causation.
- Zero covariance does not always mean independence.
- Strong correlation can be an artifact of leakage, duplicates, or confounders.
- Covariance depends on scale; correlation removes scale but not bias.

## Related

- [[concepts/math/random-variable|Random variable]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/machine-learning/linear-model|Linear model]]
- [[concepts/evaluation/leakage|Leakage]]
