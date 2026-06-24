---
title: Contrastive Learning
tags:
  - learning
  - self-supervised-learning
  - contrastive-learning
---

# Contrastive Learning

Contrastive learning trains representations by pulling related views (positives) together in embedding space and pushing unrelated views (negatives) apart. Positives are usually augmented views of the same instance; negatives are other instances.

## Why It Matters

- Learns useful representations without labels, from the structure of views alone.
- The augmentation policy encodes the invariances you want the representation to have.
- Applies to sequences, graphs, molecules, and proteins where defining a meaningful "same entity, different view" is natural.

## Design Questions

- What defines a positive pair?
- What defines a negative pair?
- Which augmentations preserve the meaning needed for downstream tasks?

## Checks

- Do augmentations preserve label-relevant content, or do they destroy it (e.g. a molecular edit that changes activity)?
- Are negatives actually negative, or do they include near-duplicates and same-family members (false negatives)?
- Does the split avoid leakage through near-duplicates, shared scaffolds, or protein families?
- Is representation collapse occurring (all embeddings converging to one point)?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/jepa|JEPA]]
- [[entities/protein|Protein]]
