---
title: Genome
aliases:
  - bio/genome
  - bio-ai/genome
tags:
  - molecular-modeling
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

## Representation Choices

| Representation | Use For | Main Risk |
| --- | --- | --- |
| Character sequence | token prediction, local classification, generation | very long context and near-duplicate leakage |
| k-mer counts or tokens | efficient baselines and sequence windows | loses long-range ordering beyond the chosen window |
| Genomic region | localization, annotation, regulatory sequence tasks | coordinate-system and reference-genome mismatch |
| Variant-centered window | variant-effect prediction | label source and train/test overlap around nearby variants |
| Annotation features | feature-augmented prediction | may encode downstream labels or source-specific artifacts |

## Task Boundary

Genome sequence modeling should state:

$$
(r,\ x_{a:b},\ c)
\rightarrow
\hat{y}
$$

where $r$ is a genomic region, $x_{a:b}$ is a sequence window, $c$ is optional annotation or organism/source context, and $\hat{y}$ is the prediction target.

## Boundary

This section does not try to cover full transcriptomics, proteomics, single-cell analysis, or multi-omics. Those can be added only if they become directly useful for a concrete research or project thread.

## Task Map

| Task | Output | Evaluation Risk |
| --- | --- | --- |
| Sequence classification | class or probability | nearby or homologous windows can leak across splits |
| Masked sequence modeling | token distribution | loss may not imply downstream biological utility |
| Variant-effect prediction | effect score or class | labels depend heavily on assay/source context |
| Annotation prediction | region label or interval | coordinate system and annotation version must match |

## Checks

- Is the input a raw sequence, annotated region, variant, or derived tokenization?
- Is the task local prediction, sequence classification, generation, or effect prediction?
- Are train/test splits preventing near-identical sequence leakage?
- Is the biological claim wider than the data source supports?

## Related

- [[molecular-modeling/index|Molecular Modeling]]
- [[ai/learning-methods|Learning methods]]
- [[concepts/modalities/sequence|Sequence]]
