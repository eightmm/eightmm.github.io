---
title: Sequence
tags:
  - entities
  - sequence
---

# Sequence

A sequence is an ordered string of tokens such as amino acids, nucleotides, or molecular text encodings.

A sequence model sees tokens, not the raw biological object:

$$
s=(t_1,\ldots,t_L),
\qquad
t_i \in \mathcal{V}
$$

where $\mathcal{V}$ is a vocabulary such as amino acids, nucleotides, k-mers, SMILES tokens, or learned units.

## Why It Matters

- Protein sequence is a compact representation for [[entities/protein|protein]] modeling.
- Sequence models can learn patterns without explicit 3D coordinates.
- Sequence-derived features are often combined with [[entities/structure|structure]] or assay labels.
- Tokenization, truncation, masking, and padding define what information reaches the model.

## Checks

- What is the tokenization: residues, atoms, SMILES tokens, k-mers, or learned units?
- Are sequence length limits, padding, and truncation affecting the task?
- Does the split avoid homolog or near-duplicate leakage?
- Is the model expected to recover structure, function, or only local sequence statistics?
- Is the sequence aligned, raw, canonicalized, segmented, or paired with metadata?

## Related

- [[entities/protein|Protein]]
- [[entities/genome|Genome]]
- [[entities/structure|Structure]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/modalities/sequence|Sequence modality]]
- [[concepts/genome-modeling/k-mer|K-mer]]
- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[entities/dataset|Dataset]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
