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

The useful representation is not necessarily the one that preserves everything. It should preserve information needed for downstream tasks and discard nuisance variation:

$$
z=f_\theta(x),
\qquad
I(z;y)\ \text{large},
\qquad
I(z;\epsilon)\ \text{small}
$$

where $y$ is task-relevant information and $\epsilon$ is nuisance variation.

## Checks

- What information should the representation preserve or discard?
- Is the representation instance-level, token-level, graph-level, or structure-level?
- Does it transfer to downstream tasks under a realistic split?
- Are embeddings evaluated with [[concepts/learning/linear-probing|linear probes]], [[concepts/learning/fine-tuning-protocol|fine-tuning]], retrieval, or task metrics?
- Does the representation avoid [[concepts/learning/representation-collapse|collapse]]?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/learning/linear-probing|Linear probing]]
- [[concepts/learning/fine-tuning-protocol|Fine-tuning protocol]]
- [[concepts/learning/representation-collapse|Representation collapse]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/architectures/embedding|Embedding]]
