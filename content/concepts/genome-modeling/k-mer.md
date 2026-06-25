---
title: K-mer
tags:
  - genome
  - sequence
  - tokenization
---

# K-mer

A k-mer is a length-$k$ substring of a biological sequence. It is a common tokenization unit for DNA, RNA, and protein-like sequence inputs.

For a sequence $x=(b_1,\ldots,b_L)$, the $i$-th k-mer is:

$$
u_i = (b_i,\ldots,b_{i+k-1})
$$

The number of overlapping k-mers is:

$$
L-k+1
$$

## Why It Matters

- K-mers convert long sequences into discrete tokens or counts.
- Larger $k$ captures more context but increases vocabulary size.
- Small $k$ can miss long-range dependencies.
- Overlapping k-mers can create near-duplicate leakage if windows are split carelessly.

## Checks

- What value of $k$ is used?
- Are ambiguous bases handled explicitly?
- Are reverse complements merged or kept separate?
- Is tokenization deterministic across train and inference?
- Does the split prevent overlapping windows from crossing train and test?

## Related

- [[concepts/architectures/tokenization|Tokenization]]
- [[entities/sequence|Sequence]]
- [[concepts/genome-modeling/genomic-region|Genomic region]]
- [[concepts/data/data-curation|Data curation]]
- [[concepts/evaluation/leakage|Leakage]]
