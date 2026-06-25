---
title: Sequence Structure Alignment
tags:
  - protein-modeling
  - data
---

# Sequence Structure Alignment

Sequence-structure alignment maps residue positions in a sequence to residues present in a structure file. It is required before combining protein language model embeddings with 3D coordinates. It depends on a clear [[concepts/protein-modeling/residue-indexing|Residue indexing]] policy.

A safe alignment defines a map:

$$
\pi: \{1,\ldots,L_{\mathrm{seq}}\}
\rightarrow
\{1,\ldots,L_{\mathrm{struct}}\}\cup\{\varnothing\}
$$

where $\pi(i)=\varnothing$ means residue $i$ is missing from the structure.

Feature fusion then uses aligned indices:

$$
z_{\pi(i)}
= [h_i, x_{\pi(i)}]
$$

only when $\pi(i)\ne\varnothing$.

## Why Raw Order Fails

The sequence index and the structure row index can diverge because of:

- missing residues
- insertion codes
- alternate locations
- unresolved loops
- engineered tags
- non-standard residues
- chain selection
- special tokens in sequence models

The failure mode is silent: tensors have compatible shapes after truncation or filtering, but residue embeddings are paired with the wrong coordinates.

## Alignment Record

A safe alignment stores:

$$
(i_{\mathrm{seq}}, r_{\mathrm{struct}}, a_{\mathrm{seq}}, a_{\mathrm{struct}}, m_i)
$$

where $a$ is the residue identity and $m_i$ records whether structure coordinates are present.

This record should be generated after [[concepts/protein-modeling/protein-structure-cleaning|Protein structure cleaning]] and before feature fusion.

## Checks

- Are missing residues, insertion codes, chain IDs, and non-standard residues handled?
- Is the FASTA sequence aligned to the structure sequence instead of using raw row order?
- Are pLM embeddings pooled or indexed with padding and special tokens excluded?
- Are coordinate units and residue numbering documented?
- Is an alignment sample inspected around missing residues and insertion codes?
- Are residues with no coordinates masked instead of shifting all later positions?

## Related

- [[concepts/protein-modeling/residue-indexing|Residue indexing]]
- [[concepts/protein-modeling/protein-structure-cleaning|Protein structure cleaning]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[entities/sequence|Sequence]]
- [[entities/structure|Structure]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/evaluation/leakage|Leakage]]
