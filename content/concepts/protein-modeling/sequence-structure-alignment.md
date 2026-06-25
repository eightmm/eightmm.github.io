---
title: Sequence Structure Alignment
tags:
  - protein-modeling
  - data
---

# Sequence Structure Alignment

Sequence-structure alignment maps residue positions in a sequence to residues present in a structure file. It is required before combining protein language model embeddings with 3D coordinates.

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

## Checks

- Are missing residues, insertion codes, chain IDs, and non-standard residues handled?
- Is the FASTA sequence aligned to the structure sequence instead of using raw row order?
- Are pLM embeddings pooled or indexed with padding and special tokens excluded?
- Are coordinate units and residue numbering documented?

## Related

- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[entities/sequence|Sequence]]
- [[entities/structure|Structure]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/evaluation/leakage|Leakage]]
