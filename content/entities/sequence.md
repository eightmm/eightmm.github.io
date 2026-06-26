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

## Sequence Contract

| Field | Question | Example |
| --- | --- | --- |
| Alphabet | What token vocabulary is used? | amino acid, nucleotide, k-mer, SMILES, byte |
| Unit | What does one example contain? | chain, domain, region, MSA row, molecule string |
| Boundary | How are chains, gaps, unknowns, and special tokens encoded? | `[UNK]`, gap, chain break, BOS/EOS |
| Length policy | Are sequences cropped, padded, chunked, or filtered? | max length, window stride, truncation side |
| Alignment | raw sequence, MSA, pair alignment, or structure alignment? | sequence-only vs aligned columns |
| Context | target, organism, assay, structure, or family metadata? | target-conditioned activity prediction |

For a protein sequence:

$$
s=(a_1,\ldots,a_L),
\qquad
a_i\in\mathcal{A}_{\mathrm{AA}}
$$

For a k-mer representation:

$$
s \rightarrow (s_{1:k}, s_{2:k+1}, \ldots, s_{L-k+1:L})
$$

The unit and tokenization decide which biological variation is visible to the model.

## Sequence vs Structure Claim

| Claim | Sequence Evidence Is Enough? | Extra Requirement |
| --- | --- | --- |
| residue language modeling | usually yes | tokenizer, masking, homolog split |
| protein family classification | often yes | family and sequence identity control |
| structure prediction | no | coordinate/contact metric and template policy |
| binding or docking | no | target, pocket, assay, or complex context |
| variant effect | not alone | variant definition, background sequence, label context |

Do not infer structural or functional understanding from sequence loss alone unless transfer evidence supports it.

## Leakage and Split

Sequence data leaks through near duplicates and homologs:

$$
\operatorname{identity}(s_i,s_j)
=
\frac{\#\text{matched residues}}{\#\text{aligned residues}}
$$

For transfer claims, the split should state whether it controls exact duplicates, sequence identity clusters, families, domains, or time.

## Checks

- What is the tokenization: residues, atoms, SMILES tokens, k-mers, or learned units?
- Are sequence length limits, padding, and truncation affecting the task?
- Does the split avoid homolog or near-duplicate leakage?
- Is the model expected to recover structure, function, or only local sequence statistics?
- Is the sequence aligned, raw, canonicalized, segmented, or paired with metadata?
- Is sequence identity computed before or after filtering and preprocessing?
- Is the sequence available at inference time without private or future context?

## Related

- [[entities/protein|Protein]]
- [[entities/genome|Genome]]
- [[entities/structure|Structure]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/modalities/sequence|Sequence modality]]
- [[concepts/genome-modeling/k-mer|K-mer]]
- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/sequence-identity-clustering|Sequence identity clustering]]
- [[entities/dataset|Dataset]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
