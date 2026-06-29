---
title: Genomic Region
tags:
  - genome
  - sequence
---

# Genomic Region

A genomic region is a contiguous interval on a genome assembly. Region-level modeling treats the interval as the input object rather than a whole genome.

A region can be represented as:

$$
r = (\text{assembly}, \text{chromosome}, s, e, \text{strand})
$$

where $s$ and $e$ are start and end coordinates.

Coordinate conventions are part of the data contract:

$$
\ell =
\begin{cases}
e-s & \text{for zero-based half-open intervals } [s,e) \\
e-s+1 & \text{for one-based closed intervals } [s,e]
\end{cases}
$$

Mixing these conventions can shift every extracted sequence by one base.

## Modeling Uses

- Sequence representation learning over fixed-length windows.
- Regulatory or functional annotation prediction.
- Variant context extraction.
- Retrieval of sequence neighborhoods.

## Region Contract

| Field | Why It Matters |
| --- | --- |
| Assembly | coordinates are only meaningful relative to a reference version |
| Chromosome / contig | naming conventions differ across sources |
| Start / end convention | off-by-one errors change the sequence |
| Strand | reverse complement may be required |
| Window rule | centered, padded, clipped, or variable-length extraction changes context |
| Overlap policy | overlapping windows can leak between splits |

## Checks

- Is the coordinate system 0-based or 1-based?
- Is the end coordinate inclusive or exclusive?
- Is the genome assembly version fixed?
- Are overlapping windows split carefully to avoid leakage?
- Does strand orientation matter for the task?
- Are regions lifted over, filtered, or padded before modeling?
- Is the split unit region, gene, chromosome, sample, family, or time?

## Related

- [[entities/genome|Genome]]
- [[concepts/genome-modeling/k-mer|K-mer]]
- [[concepts/genome-modeling/genome-annotation|Genome annotation]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
