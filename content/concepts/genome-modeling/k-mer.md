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

For an alphabet of size $|\Sigma|$, the possible vocabulary size is:

$$
|\mathcal{V}| = |\Sigma|^k
$$

This grows exponentially with $k$, so large k-mers quickly become sparse.

## Why It Matters

- K-mers convert long sequences into discrete tokens or counts.
- Larger $k$ captures more context but increases vocabulary size.
- Small $k$ can miss long-range dependencies.
- Overlapping k-mers can create near-duplicate leakage if windows are split carelessly.

## Design Choices

| Choice | Effect |
| --- | --- |
| k | controls local context and vocabulary size |
| stride | overlapping stride 1 gives dense tokens; larger stride reduces redundancy |
| reverse complement handling | merge or separate strand-equivalent tokens |
| ambiguous bases | keep, mask, discard, or map to unknown token |
| count vs sequence | bag-of-k-mers loses order; token sequence preserves local order |
| vocabulary cutoff | rare k-mers may be pruned or mapped to unknown |

## Checks

- What value of $k$ is used?
- Are ambiguous bases handled explicitly?
- Are reverse complements merged or kept separate?
- Is tokenization deterministic across train and inference?
- Does the split prevent overlapping windows from crossing train and test?
- Is the model using k-mer counts, ordered k-mer tokens, or learned embeddings?
- Is vocabulary construction fit only on training data when it depends on frequency?

## Related

- [[concepts/architectures/tokenization|Tokenization]]
- [[entities/sequence|Sequence]]
- [[concepts/genome-modeling/genomic-region|Genomic region]]
- [[concepts/data/data-curation|Data curation]]
- [[concepts/evaluation/leakage|Leakage]]
