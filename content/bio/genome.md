---
title: Genome
aliases:
  - bio-ai/genome
tags:
  - bio
  - genome
---


# Genome

Genome-level notes stay narrow in this blog. The goal is not broad omics coverage, but sequence, region, k-mer, annotation, and variant-effect concepts that may connect to representation learning or biological sequence modeling.

$$
x_{\mathrm{genome}}
\in
\{\text{sequence}, \text{region}, \text{k-mer}, \text{variant}, \text{annotation}\}
$$

## Core Notes

- [[entities/genome|Genome]]
- [[entities/sequence|Sequence]]
- [[concepts/genome-modeling/index|Genome modeling concepts]]
- [[concepts/genome-modeling/genomic-region|Genomic region]]
- [[concepts/genome-modeling/k-mer|K-mer]]
- [[concepts/genome-modeling/variant-effect-prediction|Variant effect prediction]]
- [[concepts/genome-modeling/genome-annotation|Genome annotation]]

## Boundary

This section does not try to cover full transcriptomics, proteomics, single-cell analysis, or multi-omics. Those can be added only if they become directly useful for a concrete research or project thread.

## Checks

- Is the input a raw sequence, annotated region, variant, or derived tokenization?
- Is the task local prediction, sequence classification, generation, or effect prediction?
- Are train/test splits preventing near-identical sequence leakage?
- Is the biological claim wider than the data source supports?

## Related

- [[bio/index|Bio]]
- [[ai/learning-methods|Learning methods]]
- [[concepts/modalities/sequence|Sequence]]
