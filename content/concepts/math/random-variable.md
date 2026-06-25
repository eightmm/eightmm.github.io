---
title: Random Variable
tags:
  - math
  - probability
---

# Random Variable

A random variable maps uncertain outcomes to values. In machine learning, inputs, labels, predictions, latent variables, noise, and evaluation scores can all be treated as random variables.

Formally, a random variable is a function:

$$
X: \Omega \to \mathcal{X}
$$

where $\Omega$ is the sample space and $\mathcal{X}$ is the value space.

## Distribution

The distribution of $X$ describes how likely each value is:

$$
P(X \in A)
$$

for a set of possible values $A \subseteq \mathcal{X}$.

For a discrete variable:

$$
P(X=x)=p(x)
$$

For a continuous variable:

$$
P(a \le X \le b)
= \int_a^b p(x)\,dx
$$

where $p(x)$ is a probability mass or density function depending on the variable type.

## Why It Matters

- A dataset is a finite sample from random variables $(X,Y)$.
- A model output can estimate $p(Y\mid X=x)$.
- A loss is often a random variable before it is averaged.
- Metrics estimate expected behavior from finite samples.

## Checks

- What is random: input, label, noise, model output, split, or metric?
- What distribution is assumed: training, validation, test, deployment, or model distribution?
- Is the variable discrete, continuous, vector-valued, graph-valued, or structured?
- Is the observed dataset one sample, or a distribution itself?

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/evaluation/metric|Metric]]
