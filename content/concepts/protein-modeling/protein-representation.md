---
title: Protein Representation
tags:
  - protein-modeling
  - representation-learning
---

# Protein Representation

A protein representation is the form of a protein used by a model: sequence tokens, embeddings, residue graphs, contact maps, coordinates, surfaces, or mixed sequence-structure features.

For a sequence $s_{1:L}$, a representation model maps residues to hidden states:

$$
H = f_\theta(s_{1:L}),
\qquad
H\in\mathbb{R}^{L\times d}
$$

A protein-level embedding can be formed by masked pooling:

$$
h_{\mathrm{protein}}
= \frac{\sum_{i=1}^{L} m_i h_i}
{\sum_{i=1}^{L} m_i}
$$

where $m_i$ is an attention mask that excludes padding and special tokens.

## Common Views

- Residue sequence for language-model pretraining.
- MSA profile for evolutionary coupling.
- Residue graph or contact map for structural relationships.
- 3D coordinates for geometry-aware modeling.
- Pocket representation for structure-based interaction tasks.

## Checks

- What does one token or node represent?
- Are padding, truncation, chain breaks, and non-standard residues handled?
- Is the representation invariant, equivariant, or frame-dependent?
- Is sequence-structure alignment verified before fusing embeddings with coordinates?

## Related

- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/protein-modeling/contact-map|Contact map]]
- [[concepts/protein-modeling/sequence-structure-alignment|Sequence-structure alignment]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[research/protein-modeling/index|Protein modeling]]
