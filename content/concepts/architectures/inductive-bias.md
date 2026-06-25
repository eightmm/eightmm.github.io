---
title: Inductive bias
tags:
  - architectures
  - machine-learning
---

# Inductive Bias

Inductive bias is the assumption a model family uses to prefer some functions over others before seeing the full data distribution.

An architecture restricts the hypothesis class:

$$
f_\theta \in \mathcal{F}_{\mathrm{arch}}
$$

Training then searches within that class:

$$
\hat f
= \arg\min_{f \in \mathcal{F}_{\mathrm{arch}}}
\frac{1}{n}\sum_{i=1}^{n}\ell(f(x_i), y_i)
$$

Here $x_i$ is an input, $y_i$ is a target, $\ell$ is the loss, and $\mathcal{F}_{\mathrm{arch}}$ is the function class implied by the architecture.

## Common Biases

- [[concepts/architectures/cnn|CNN]]: locality and translation equivariance.
- [[concepts/architectures/rnn|RNN]] and [[concepts/architectures/state-space-model|state-space models]]: ordered state updates over a sequence.
- [[concepts/architectures/transformer|Transformer]]: content-based pairwise interaction through attention.
- [[concepts/architectures/gnn|GNN]]: graph connectivity and permutation equivariance over nodes.
- [[concepts/geometric-deep-learning/equivariance|Equivariant models]]: output transforms predictably under rotation, translation, or other symmetry groups.
- [[concepts/architectures/mixture-of-experts|Mixture of experts]]: sparse conditional computation through routing.

## Why It Matters

The right bias can reduce sample complexity because the model does not need to learn every useful symmetry from data. The wrong bias can hide important interactions or make a benchmark look better than deployment.

For example, a graph model is natural when edges encode meaningful relations. If the graph construction is noisy or leaks target information, the architecture can amplify the wrong signal.

## Checks

- What symmetry or invariance does the data actually have?
- Is the architecture assuming locality, order, graph structure, sparsity, or geometry?
- Does the benchmark split test the same bias that deployment will require?
- Is a simpler model with a weaker bias a necessary baseline?

## Related

- [[concepts/architectures/index|Architectures]]
- [[concepts/architectures/parameter-sharing|Parameter sharing]]
- [[concepts/architectures/architecture-selection|Architecture selection]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/baseline|Baseline]]
