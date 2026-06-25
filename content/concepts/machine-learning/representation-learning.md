---
title: Representation Learning
tags:
  - machine-learning
  - representation-learning
---

# Representation Learning

Representation learning trains a model to transform raw inputs into features that make downstream tasks easier.

A representation model maps inputs to embeddings:

$$
z = f_\theta(x),
\qquad
z\in\mathbb{R}^{d}
$$

A downstream model then predicts from $z$:

$$
\hat{y}=g_\phi(z)
$$

The representation can be learned with supervised labels, self-supervised signals, contrastive objectives, masked modeling, or generative pretraining.

## Checks

- What information should the representation preserve or discard?
- Is the representation instance-level, token-level, graph-level, or structure-level?
- Does it transfer to downstream tasks under a realistic split?
- Are embeddings evaluated with linear probes, fine-tuning, retrieval, or task metrics?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/architectures/embedding|Embedding]]
