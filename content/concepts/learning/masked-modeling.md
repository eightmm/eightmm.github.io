---
title: Masked Modeling
tags:
  - masked-modeling
  - self-supervised-learning
  - representation-learning
---

# Masked Modeling

Masked modeling hides part of an input and trains the model to predict the missing content from the visible context. The masked target can be tokens, patches, nodes, or residues.

The typical objective is:

$$
\mathcal{L}
= -\sum_{i\in M}
\log p_\theta(x_i \mid x_{\setminus M})
$$

Here $M$ is the masked subset and $x_{\setminus M}$ is the visible context.

For continuous targets, the reconstruction loss may be:

$$
\mathcal{L}
=
\sum_{i\in M}
\lVert \hat{x}_i - x_i\rVert_2^2
$$

where $\hat{x}_i$ is the model's prediction for the masked element.

## Masking Design

- Random token or patch masking tests local context recovery.
- Span masking forces longer-range reasoning.
- Node or edge masking tests graph context.
- Residue or atom masking tests biological or chemical context.
- Coordinate or distance masking tests geometric consistency.

## Why It Matters

- A simple, scalable self-supervised objective across text, images, graphs, and sequences.
- Builds context-aware representations without manual labels.
- Underlies many pretrained encoders for language, vision, and biomolecules.
- The masking pattern determines whether the task is trivial, useful, or impossible.

## Checks

- Does the masking ratio and pattern match the data's redundancy?
- Is the prediction target reconstructed in pixel/token space or in a latent space?
- Could trivial shortcuts solve the pretext task without learning structure?
- Are masked positions excluded from the visible input and metadata?
- Does the objective learn reusable representations, or only local reconstruction tricks?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
