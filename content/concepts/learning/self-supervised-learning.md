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

More explicitly, many SSL methods sample one or more views:

$$
v_1, v_2 \sim \mathcal{A}(x)
$$

where $\mathcal{A}$ is an [[concepts/learning/augmentation-policy|augmentation policy]]. The method then defines which information should match between views and which information can be ignored.

## Objective Families

- Masked prediction: hide part of the input and predict the missing target.
- Contrastive learning: align positive views while separating negatives.
- Joint-embedding prediction: predict target embeddings from context embeddings.
- Autoregressive prediction: predict future or next tokens from previous context.
- Reconstruction or denoising: recover a clean input from a corrupted one.

## Why It Matters

- Useful when labels are sparse or expensive.
- Can pretrain sequence, graph, molecular, and protein representations.
- Often shifts the main question from architecture to data construction and evaluation.
- Makes data curation and leakage control part of the learning method, not only preprocessing.

## Evaluation

SSL is usually evaluated indirectly through downstream performance:

$$
z=f_\theta(x),
\qquad
\hat{y}=g_\phi(z)
$$

Common evaluation modes are [[concepts/learning/linear-probing|linear probing]], [[concepts/learning/fine-tuning-protocol|full fine-tuning]], retrieval, clustering quality, and task-specific metrics. These belong to [[concepts/learning/representation-evaluation|representation evaluation]], not to the pretraining objective itself.

## Checks

- Is the pretext task aligned with the downstream task?
- Can train and test data leak through near-duplicates, scaffolds, protein families, or temporal splits?
- Does the representation transfer beyond the pretraining distribution?
- Does the objective avoid [[concepts/learning/representation-collapse|representation collapse]]?
- Are augmentation and masking choices valid for the domain?
- Are linear probe, fine-tuning, and retrieval protocols kept separate?

## Related

- [[ai/learning-methods|Learning methods]]
- [[concepts/learning/pretraining|Pretraining]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/learning/representation-collapse|Representation collapse]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/learning/linear-probing|Linear probing]]
- [[concepts/learning/fine-tuning-protocol|Fine-tuning protocol]]
- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
