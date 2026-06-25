---
title: Sequence
tags:
  - entities
  - sequence
---

# Sequence

A sequence is an ordered string of tokens such as amino acids, nucleotides, or molecular text encodings.

## Why It Matters

- Protein sequence is a compact representation for [[entities/protein|protein]] modeling.
- Sequence models can learn patterns without explicit 3D coordinates.
- Sequence-derived features are often combined with [[entities/structure|structure]] or assay labels.

## Checks

- What is the tokenization: residues, atoms, SMILES tokens, k-mers, or learned units?
- Are sequence length limits, padding, and truncation affecting the task?
- Does the split avoid homolog or near-duplicate leakage?
- Is the model expected to recover structure, function, or only local sequence statistics?

## Related

- [[entities/protein|Protein]]
- [[entities/structure|Structure]]
- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[entities/dataset|Dataset]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
