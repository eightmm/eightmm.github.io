---
title: Self-Supervised Learning
tags:
  - self-supervised-learning
  - representation-learning
  - machine-learning
---

# Self-Supervised Learning

Self-supervised learning trains representations from structure in the data instead of direct human labels. The target can be masked tokens, corrupted views, contrastive pairs, future states, or reconstruction objectives.

## Why It Matters

- Useful when labels are sparse or expensive.
- Can pretrain sequence, graph, molecular, and protein representations.
- Often shifts the main question from architecture to data construction and evaluation.

## Checks

- Is the pretext task aligned with the downstream task?
- Can train and test data leak through near-duplicates, scaffolds, protein families, or temporal splits?
- Does the representation transfer beyond the pretraining distribution?

## Related

- [[research/self-supervised-learning/index|Self-supervised learning]]
- [[research/protein-modeling/index|Protein modeling]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
