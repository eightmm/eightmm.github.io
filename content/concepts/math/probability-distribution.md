---
title: Probability Distribution
tags:
  - math
  - probability
---

# Probability Distribution

A probability distribution assigns probabilities to possible outcomes. In machine learning, data, labels, latent variables, model outputs, noise, and deployment environments are often described as distributions.

For a discrete variable:

$$
\sum_x p(x) = 1,\qquad p(x)\ge 0
$$

For a continuous variable:

$$
\int p(x)\,dx = 1,\qquad p(x)\ge 0
$$

## Joint, Marginal, and Conditional

Joint distribution:

$$
p(x,y)
$$

Marginal distribution:

$$
p(x)=\sum_y p(x,y)
$$

Conditional distribution:

$$
p(y\mid x)=\frac{p(x,y)}{p(x)}
$$

## Checks

- Is the distribution over inputs, labels, latents, outputs, or noise?
- Is the model estimating $p(x)$, $p(y\mid x)$, $p(x,y)$, or a score?
- Is the variable discrete, continuous, structured, or mixed?
- Does the training distribution match the deployment distribution?

## Related

- [[concepts/math/expectation|Expectation]]
- [[concepts/math/bayes-rule|Bayes rule]]
- [[concepts/machine-learning/density-estimation|Density estimation]]
- [[concepts/generative-models/index|Generative models]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
