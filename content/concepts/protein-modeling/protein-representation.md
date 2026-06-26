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

## Sequence Representation

A sequence-only model treats residues as tokens:

$$
s_{1:L}
\rightarrow
e_{1:L}
\rightarrow
h_{1:L}
$$

where $e_i$ is a residue embedding and $h_i$ is a contextual representation. This is the common setup for protein language models. It captures sequence statistics and evolutionary signals indirectly, but it does not explicitly encode 3D coordinates.

## Structure Representation

A structure representation usually contains residue or atom coordinates:

$$
X
=
[x_1,\dots,x_N],
\qquad
x_i\in\mathbb{R}^3
$$

If a model consumes coordinates, the transformation behavior must be clear. Scalar predictions such as binding labels should be invariant to rigid transforms:

$$
f(RX+t)=f(X)
$$

Vector or coordinate outputs should usually be equivariant:

$$
f(RX+t)=R f(X)+t
$$

where $R$ is a rotation and $t$ is a translation.

## Graph Representation

A residue graph can be written as:

$$
G=(V,E),
\qquad
E=\{(i,j): \lVert x_i-x_j\rVert_2 < \tau\}
$$

Edges may come from sequence adjacency, spatial cutoff, k-nearest neighbors, contact maps, or learned attention. The graph policy changes what the model can see.

## Fusion Risk

Sequence-structure fusion aligns sequence embeddings with structure residues:

$$
(h_i, x_i)
\quad
\text{for residue } i
$$

This alignment is fragile. Missing density, insertion codes, chain breaks, alternate residue numbering, and non-standard residues can silently pair the wrong embedding with the wrong coordinate.

## Common Views

- Residue sequence for language-model pretraining.
- MSA profile for evolutionary coupling.
- Residue graph or contact map for structural relationships.
- 3D coordinates for geometry-aware modeling.
- Pocket representation for structure-based interaction tasks.
- Sequence-structure fused representation using aligned embeddings and coordinates.

## Checks

- What does one token or node represent?
- Are padding, truncation, chain breaks, and non-standard residues handled?
- Is the representation invariant, equivariant, or frame-dependent?
- Is sequence-structure alignment verified before fusing embeddings with coordinates?
- Is residue indexing preserved after structure cleaning and feature caching?
- Does the representation include full protein, domain, pocket, chain, residue, or atom context?
- Are train-time and inference-time coordinate sources matched, for example crystal structures vs predicted structures?
- Does batching preserve per-protein isolation for pooling, normalization, and attention?

## Related

- [[concepts/protein-modeling/residue-indexing|Residue indexing]]
- [[concepts/protein-modeling/protein-structure-cleaning|Protein structure cleaning]]
- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/protein-modeling/contact-map|Contact map]]
- [[concepts/protein-modeling/sequence-structure-alignment|Sequence-structure alignment]]
- [[concepts/protein-modeling/pocket-representation|Pocket representation]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[research/protein-modeling/index|Protein modeling]]
