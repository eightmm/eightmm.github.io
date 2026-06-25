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

## Core Notes

- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/math/bayes-rule|Bayes rule]]

## Checks

- What object is the formula about: scalar, vector, matrix, graph, distribution, or function?
- Is the quantity optimized, estimated, sampled, normalized, or measured?
- Is a probability statement conditional or marginal?
- Is a loss a negative log-likelihood, distance, ranking loss, or surrogate?
- Are expectations taken under data, model, sampling, or deployment distributions?
- If the object has coordinates, what transformations should preserve meaning?

## Related

- [[concepts/machine-learning/index|Machine learning]]
- [[concepts/generative-models/index|Generative models]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/architectures/index|Architectures]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
