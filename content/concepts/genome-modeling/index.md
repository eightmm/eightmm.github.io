---
title: Genome Modeling Concepts
tags:
  - genome
  - sequence
  - molecular-modeling
---

# Genome Modeling Concepts

Genome modeling concepts in this wiki are limited to sequence-level and variant-level AI. They are included as a boundary topic for Molecular Modeling, not as a broad omics or transcriptomics section.

The basic input is a genomic sequence or region:

$$
x = (b_1,\ldots,b_L), \qquad b_i \in \{A,C,G,T,N\}
$$

The model may predict a representation, annotation, variant effect, or generated sequence:

$$
\hat{y}=f_\theta(x, c)
$$

where $c$ is optional context such as species, genome assembly, region type, or annotation source.

## Core Concepts

- [[entities/genome|Genome]]
- [[concepts/genome-modeling/genomic-region|Genomic region]]
- [[concepts/genome-modeling/k-mer|K-mer]]
- [[concepts/genome-modeling/variant-effect-prediction|Variant effect prediction]]
- [[concepts/genome-modeling/genome-annotation|Genome annotation]]

## Checks

- What genome assembly and coordinate system are used?
- What is one example: base, k-mer, region, gene, variant, or chromosome segment?
- Are homologous, duplicated, or overlapping regions separated across splits?
- Are labels public, reproducible, and tied to their annotation source?
- Is the task sequence-level, region-level, variant-level, or annotation-level?

## Boundary

This section intentionally does not expand into broad cell omics, transcriptomics, or clinical genomics. Those topics should only be added if they become directly useful for the blog's research direction.

## Related

- [[molecular-modeling/index|Molecular Modeling]]
- [[entities/sequence|Sequence]]
- [[concepts/modalities/text|Text]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/evaluation/leakage|Leakage]]
