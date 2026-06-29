---
title: Genome Sequence Modeling
aliases:
  - computational-biology/genome
  - computational-biology/genome-sequence-modeling
  - bio/genome
tags:
  - computational-biology
  - genome
---


# Genome Sequence Modeling

Genome-level note는 이 블로그에서 좁게 유지합니다. 목표는 broad omics coverage가 아니라 representation learning 또는 biological sequence modeling과 연결될 수 있는 sequence, region, k-mer, annotation, variant-effect 개념을 정리하는 것입니다.

This page is the route for genome sequence modeling. The object definition lives in [[entities/genome|Genome]]; this page is about how genome sequence, region, k-mer, annotation, or variant context becomes a modeling problem.

$$
x_{\mathrm{genome}}
\in
\{\text{sequence}, \text{region}, \text{k-mer}, \text{variant}, \text{annotation}\}
$$

이 페이지는 현재 블로그 범위에서 genome을 “sequence modeling object”로 다룹니다. Omics-wide analysis나 clinical pipeline이 아니라, AI representation과 연결되는 sequence, window, variant, annotation 문제에 초점을 둡니다.

## Route Map

| 질문 | 시작점 | 주의점 |
| --- | --- | --- |
| sequence unit이 무엇인가? | [Genome](/entities/genome), [Sequence](/entities/sequence), [Genomic region](/concepts/genome-modeling/genomic-region) | reference coordinate mismatch와 duplicated window |
| sequence를 어떻게 표현하는가? | [K-mer](/concepts/genome-modeling/k-mer), [Genome modeling concepts](/concepts/genome-modeling) | window length와 context truncation |
| variant-centered task인가? | [Variant effect prediction](/concepts/genome-modeling/variant-effect-prediction) | label source, nearby variant leakage, assay scope |
| annotation이 input인가 target인가? | [Genome annotation](/concepts/genome-modeling/genome-annotation) | annotation version과 label leakage |

## Unit Choices

| Unit | 의미 | Risk |
| --- | --- | --- |
| Whole chromosome | 매우 긴 reference sequence | model input으로는 보통 너무 넓음 |
| Fixed window | coordinate 주변 sequence slice | duplicated 또는 overlapping window |
| Region | biological annotation이 붙은 interval | coordinate system mismatch |
| Variant-centered window | reference/alternate context | nearby variant와 label-source leakage |
| k-mer stream | 짧은 subsequence token stream | long-range relation 손실 |

## Representation Choices

| Representation | 쓰임 | 주요 Risk |
| --- | --- | --- |
| Character sequence | token prediction, local classification, generation | very long context와 near-duplicate leakage |
| k-mer counts or tokens | efficient baseline과 sequence window | chosen window 밖 long-range ordering 손실 |
| Genomic region | localization, annotation, regulatory sequence task | coordinate-system과 reference-genome mismatch |
| Variant-centered window | variant-effect prediction | label source와 nearby variant train/test overlap |
| Annotation features | feature-augmented prediction | downstream label 또는 source-specific artifact를 encode할 수 있음 |

## Task Boundary

Genome sequence modeling should state:

$$
(r,\ x_{a:b},\ c)
\rightarrow
\hat{y}
$$

여기서 $r$은 genomic region, $x_{a:b}$는 sequence window, $c$는 optional annotation 또는 organism/source context, $\hat{y}$는 prediction target입니다.

## Coordinate Contract

Genome note는 coordinate assumption을 명시적으로 기록해야 합니다.

| Field | 이유 |
| --- | --- |
| reference genome | coordinate는 reference에 상대적으로만 의미가 있음 |
| strand | reverse-complement handling이 input을 바꿈 |
| interval convention | 0-based/1-based와 inclusive/exclusive mismatch가 label을 깨뜨림 |
| window policy | padding, truncation, overlap, center variant |
| annotation version | label과 feature source drift |

## Boundary

이 섹션은 full transcriptomics, proteomics, single-cell analysis, multi-omics를 다루려는 곳이 아닙니다. 그런 주제는 구체적인 연구나 프로젝트 thread에 직접 필요해질 때만 추가합니다.

## Task Map

| Task | Output | Evaluation Risk |
| --- | --- | --- |
| Sequence classification | class 또는 probability | nearby 또는 homologous window가 split 사이로 새어 나갈 수 있음 |
| Masked sequence modeling | token distribution | loss가 downstream biological utility를 뜻하지 않을 수 있음 |
| Variant-effect prediction | effect score 또는 class | label이 assay/source context에 크게 의존 |
| Annotation prediction | region label 또는 interval | coordinate system과 annotation version이 맞아야 함 |

## Checks

- input이 raw sequence, annotated region, variant, derived tokenization 중 무엇인가?
- task가 local prediction, sequence classification, generation, effect prediction 중 무엇인가?
- train/test split이 near-identical sequence leakage를 막는가?
- biological claim이 data source가 지지하는 범위보다 넓지 않은가?

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[entities/genome|Genome]]
- [[ai/learning-methods|Learning methods]]
- [[concepts/modalities/sequence|Sequence]]
