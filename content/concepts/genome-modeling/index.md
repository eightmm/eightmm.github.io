---
title: Genome Modeling Concepts
tags:
  - genome
  - sequence
  - molecular-modeling
---

# Genome Modeling Concepts

이 wiki의 genome modeling concept는 sequence-level과 variant-level AI로 제한합니다. Broad omics나 transcriptomics 섹션이 아니라 Computational Biology의 boundary topic으로 둡니다.

기본 input은 genomic sequence 또는 region입니다.

$$
x = (b_1,\ldots,b_L), \qquad b_i \in \{A,C,G,T,N\}
$$

The model may predict a representation, annotation, variant effect, or generated sequence:

$$
\hat{y}=f_\theta(x, c)
$$

where $c$ is optional context such as species, genome assembly, region type, or annotation source.

## Route Map

| Need | Start | Risk |
| --- | --- | --- |
| define the object | [Genome](/entities/genome), [Genomic region](/concepts/genome-modeling/genomic-region) | reference assembly and coordinate-system mismatch |
| tokenize sequence | [K-mer](/concepts/genome-modeling/k-mer) | window overlap and loss of long-range context |
| predict variant impact | [Variant effect prediction](/concepts/genome-modeling/variant-effect-prediction) | assay/source-dependent labels |
| use annotations | [Genome annotation](/concepts/genome-modeling/genome-annotation) | annotation version and target leakage |
| connect to learning | [Self-supervised learning](/concepts/learning/self-supervised-learning), [Sequence](/concepts/modalities/sequence) | pretraining loss may not support downstream claim |

## Core Concepts

| Group | Notes |
| --- | --- |
| Objects | [Genome](/entities/genome), [Sequence](/entities/sequence), [Genomic region](/concepts/genome-modeling/genomic-region) |
| Representation | [K-mer](/concepts/genome-modeling/k-mer), [Sequence modality](/concepts/modalities/sequence) |
| Tasks | [Variant effect prediction](/concepts/genome-modeling/variant-effect-prediction), [Genome annotation](/concepts/genome-modeling/genome-annotation) |

## Checks

- What genome assembly and coordinate system are used?
- What is one example: base, k-mer, region, gene, variant, or chromosome segment?
- Are homologous, duplicated, or overlapping regions separated across splits?
- Are labels public, reproducible, and tied to their annotation source?
- Is the task sequence-level, region-level, variant-level, or annotation-level?

## Boundary

이 섹션은 의도적으로 broad cell omics, transcriptomics, clinical genomics로 확장하지 않습니다. 그런 주제는 블로그의 research direction에 직접 필요해질 때만 추가합니다.

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[entities/sequence|Sequence]]
- [[concepts/modalities/text|Text]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/evaluation/leakage|Leakage]]
