---
title: Contrastive Learning
tags:
  - learning
  - self-supervised-learning
  - contrastive-learning
---

# Contrastive Learning

Contrastive learning trains representations by pulling related views (positives) together in embedding space and pushing unrelated views (negatives) apart. Positives are usually augmented views of the same instance; negatives are other instances.

The common InfoNCE form is:

$$
\mathcal{L}_i
= -\log
\frac{\exp(\operatorname{sim}(z_i,z_i^+)/\tau)}
\sum_{j} \exp(\operatorname{sim}(z_i,z_j)/\tau)}
$$

Here $z_i^+$ is a positive view, $z_j$ are candidate positives/negatives, $\operatorname{sim}$ is a similarity function, and $\tau$ is temperature.

With normalized embeddings, cosine similarity is:

$$
\operatorname{sim}(z_i,z_j)
=
\frac{z_i^\top z_j}
{\lVert z_i\rVert_2\lVert z_j\rVert_2}
$$

The temperature $\tau$ controls how sharply the model distinguishes close and far examples.

## Why It Matters

- Learns useful representations without labels, from the structure of views alone.
- The augmentation policy encodes the invariances you want the representation to have.
- Applies to sequences, graphs, molecules, and proteins where defining a meaningful "same entity, different view" is natural.

## Design Questions

- What defines a positive pair?
- What defines a negative pair?
- Which augmentations preserve the meaning needed for downstream tasks?
- Is the objective instance-discrimination, supervised contrastive learning, cross-modal alignment, or retrieval training?

## Positive and Negative Semantics

For an anchor $x_i$, a positive should preserve the relevant identity:

$$
y(x_i) = y(x_i^+)
$$

for the intended downstream meaning $y$. A false negative occurs when $x_j$ is treated as negative even though it shares the relevant class, scaffold, family, target, or semantic state.

## Checks

- Do augmentations preserve label-relevant content, or do they destroy it (e.g. a molecular edit that changes activity)?
- Are negatives actually negative, or do they include near-duplicates and same-family members (false negatives)?
- Does the split avoid leakage through near-duplicates, shared scaffolds, or protein families?
- Is representation collapse occurring (all embeddings converging to one point)?
- Does the batch or memory bank contain enough hard negatives without contaminating the split?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/learning/representation-collapse|Representation collapse]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/evaluation/leakage|Leakage]]
- [[entities/protein|Protein]]
