---
title: Genome
aliases:
  - computational-biology/genome
  - bio/genome
tags:
  - computational-biology
  - genome
---


# Genome

Genome-level notes stay narrow in this blog. The goal is not broad omics coverage, but sequence, region, k-mer, annotation, and variant-effect concepts that may connect to representation learning or biological sequence modeling.

$$
x_{\mathrm{genome}}
\in
\{\text{sequence}, \text{region}, \text{k-mer}, \text{variant}, \text{annotation}\}
$$

이 페이지는 현재 블로그 범위에서 genome을 “sequence modeling object”로 다룹니다. Omics-wide analysis나 clinical pipeline이 아니라, AI representation과 연결되는 sequence/window/variant/annotation 문제에 초점을 둡니다.

## Route Map

| Question | Start | Watch |
| --- | --- | --- |
| What is the sequence unit? | [Genome](/entities/genome), [Sequence](/entities/sequence), [Genomic region](/concepts/genome-modeling/genomic-region) | reference coordinate mismatch and duplicated windows |
| How is the sequence represented? | [K-mer](/concepts/genome-modeling/k-mer), [Genome modeling concepts](/concepts/genome-modeling) | window length and context truncation |
| Is the task variant-centered? | [Variant effect prediction](/concepts/genome-modeling/variant-effect-prediction) | label source, nearby variant leakage, assay scope |
| Are annotations inputs or targets? | [Genome annotation](/concepts/genome-modeling/genome-annotation) | annotation version and label leakage |

## Unit Choices

| Unit | Meaning | Risk |
| --- | --- | --- |
| Whole chromosome | very long reference sequence | usually too broad for model input |
| Fixed window | sequence slice around coordinate | duplicated or overlapping windows |
| Region | interval with biological annotation | coordinate system mismatch |
| Variant-centered window | reference/alternate context | nearby variant and label-source leakage |
| k-mer stream | tokenized short subsequences | loses long-range relation |

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

## Coordinate Contract

Genome note should record coordinate assumptions explicitly.

| Field | Why |
| --- | --- |
| reference genome | coordinates only make sense relative to a reference |
| strand | reverse-complement handling changes input |
| interval convention | 0-based/1-based and inclusive/exclusive mismatch breaks labels |
| window policy | padding, truncation, overlap, center variant |
| annotation version | label and feature source drift |

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

- [[molecular-modeling/index|Computational Biology]]
- [[ai/learning-methods|Learning methods]]
- [[concepts/modalities/sequence|Sequence]]
