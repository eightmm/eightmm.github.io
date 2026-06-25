---
title: Self-Supervised Learning
tags:
  - research
  - self-supervised-learning
---

# Self-Supervised Learning Research

Self-supervised learning notes cover representation learning without direct task labels. For this site, the focus is protein, molecule, graph, and sequence representations.

The research pattern is to pretrain a representation and test transfer:

$$
z = f_\theta(x),
\qquad
\hat{y} = g_\phi(z)
$$

The key question is whether $z$ captures information that generalizes beyond the pretraining distribution.

## Topics

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]

## Questions

- Which pretext objective matches the downstream task?
- What leakage can enter through sequence, scaffold, protein family, or benchmark splits?
- How should representations be evaluated outside the pretraining distribution?

## Related

- [[research/protein-modeling/index|Protein modeling]]
- [[research/structure-based-ai/index|Structure-based AI]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
