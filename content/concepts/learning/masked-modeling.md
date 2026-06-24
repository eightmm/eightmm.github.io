---
title: Masked Modeling
tags:
  - masked-modeling
  - self-supervised-learning
  - representation-learning
---

# Masked Modeling

Masked modeling hides part of an input and trains the model to predict the missing content from the visible context. The masked target can be tokens, patches, nodes, or residues.

## Why It Matters

- A simple, scalable self-supervised objective across text, images, graphs, and sequences.
- Builds context-aware representations without manual labels.
- Underlies many pretrained encoders for language, vision, and biomolecules.

## Checks

- Does the masking ratio and pattern match the data's redundancy?
- Is the prediction target reconstructed in pixel/token space or in a latent space?
- Could trivial shortcuts solve the pretext task without learning structure?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
