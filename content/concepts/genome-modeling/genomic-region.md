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

## Modeling Uses

- Sequence representation learning over fixed-length windows.
- Regulatory or functional annotation prediction.
- Variant context extraction.
- Retrieval of sequence neighborhoods.

## Checks

- Is the coordinate system 0-based or 1-based?
- Is the end coordinate inclusive or exclusive?
- Is the genome assembly version fixed?
- Are overlapping windows split carefully to avoid leakage?
- Does strand orientation matter for the task?

## Related

- [[entities/genome|Genome]]
- [[concepts/genome-modeling/k-mer|K-mer]]
- [[concepts/genome-modeling/genome-annotation|Genome annotation]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
