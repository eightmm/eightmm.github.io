---
title: Self-Supervised Learning
tags:
  - self-supervised-learning
  - representation-learning
  - machine-learning
---

# Self-Supervised Learning

Self-supervised learning trains representations from structure in the data instead of direct human labels. The target can be masked tokens, corrupted views, contrastive pairs, future states, or reconstruction objectives.

The common pattern is to create a training signal from the input:

$$
t = T(x),
\qquad
\min_\theta \mathcal{L}(f_\theta(V(x)), t)
$$

Here $V(x)$ is the visible or augmented view and $T(x)$ constructs a target from the same raw example.

## Why It Matters

- Useful when labels are sparse or expensive.
- Can pretrain sequence, graph, molecular, and protein representations.
- Often shifts the main question from architecture to data construction and evaluation.

## Checks

- Is the pretext task aligned with the downstream task?
- Can train and test data leak through near-duplicates, scaffolds, protein families, or temporal splits?
- Does the representation transfer beyond the pretraining distribution?

## Related

- [[ai/learning-methods|Learning methods]]
- [[research/protein-modeling/index|Protein modeling]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
