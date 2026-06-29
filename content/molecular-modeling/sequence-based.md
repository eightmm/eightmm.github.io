---
title: Sequence-Based Modeling
aliases:
  - computational-biology/sequence-based
  - bio/sequence-based
tags:
  - computational-biology
  - sequence-modeling
---

# Sequence-Based Modeling

Sequence-based modeling은 biological string을 primary input으로 다룹니다. 이 wiki의 주 대상은 protein sequence와 좁은 genome-level sequence task입니다. Broad omics, transcriptomics, single-cell analysis, clinical biology는 실제 연구나 프로젝트 필요가 생길 때 별도로 엽니다.

$$
\hat{y}
=
f_\theta(s_{1:L}, c)
$$

여기서 $s_{1:L}$은 token sequence이고, $c$는 organism, target family, region, assay, mutation, annotation source 같은 context입니다.

## Route Map

| Question | Start | Watch |
| --- | --- | --- |
| 어떤 sequence object를 모델링하는가? | [Objects and Entities](/molecular-modeling/entities), [Sequence](/entities/sequence) | sequence가 protein chain, genome window, variant context, derived token stream일 수 있음 |
| protein sequence modeling인가? | [Proteins](/molecular-modeling/proteins), [Protein](/entities/protein) | homolog leakage, isoform choice, construct, mutation, truncation |
| genome-level sequence modeling인가? | [Genome](/molecular-modeling/genome), [Genome](/entities/genome) | reference coordinate mismatch, duplicated window, annotation leakage |
| representation이 learned인가 fixed인가? | [Embedding](/concepts/architectures/embedding), [Tokenization](/concepts/architectures/tokenization) | pooling과 token policy가 claim을 바꿀 수 있음 |
| 어떤 split이 claim을 지지하는가? | [Protein family split](/concepts/evaluation/protein-family-split), [Leakage](/concepts/evaluation/leakage) | random row split이 generalization을 과장할 수 있음 |

## Main Subroutes

| Area | Use for | Start |
| --- | --- | --- |
| Protein sequence | protein language models, mutation effect, family-aware representation, sequence-to-function | [Proteins](/molecular-modeling/proteins) |
| Genome sequence | genome window, k-mer, annotation, variant-effect prediction | [Genome](/molecular-modeling/genome) |
| Sequence representation | tokenization, embedding, pooling, sequence length, context window | [Sequence](/concepts/modalities/sequence) |
| Sequence evaluation | family split, near-duplicate control, label-source boundary | [Data and Evaluation](/molecular-modeling/data-evaluation) |

## Sequence Representation Checklist

| Choice | 중요한 이유 |
| --- | --- |
| Token unit | amino acid, k-mer, BPE-like token, nucleotide, special token |
| Context length | truncation이 domain, motif, regulatory context를 제거할 수 있음 |
| Pooling | CLS, mean pooling, residue pooling, region pooling은 서로 다른 claim을 지지함 |
| Alignment input | MSA, template, annotation, family information이 test context를 leak할 수 있음 |
| Mutation policy | wild-type, mutant, delta representation, paired sequence가 task를 바꿈 |

## Sequence vs Structure

Sequence-based modeling은 structure-based modeling의 반대가 아닙니다. 많은 workflow는 sequence에서 시작해 나중에 predicted 또는 experimental structure를 사용합니다.

$$
s_{1:L}
\rightarrow
h_{1:L}
\rightarrow
X
\rightarrow
\hat{y}
$$

Primary input이 sequence이면 이 페이지에서 시작합니다. Coordinate, pocket, pose, complex가 first-class object가 되면 [[molecular-modeling/structure-based/index|Structure-based modeling]]으로 이동합니다.

## Boundary

- Sequence-based: sequence token, mutation, region, family split, sequence representation이 claim의 중심입니다.
- Structure-based: coordinate, pocket, pose, conformer, complex geometry가 claim의 중심입니다.
- AI: Transformer, SSM, GNN, SSL 같은 architecture나 learning method 자체가 claim의 중심입니다.
- Math: likelihood, embedding similarity, gradient, geometry 식 자체를 설명합니다.

## Checks

- one example이 protein chain, sequence window, variant-centered window, annotated region 중 무엇인가?
- homolog, near duplicate, repeated window가 train/test 사이에서 분리되어 있는가?
- model이 inference time에 unavailable한 structure, template, MSA, annotation을 쓰는가?
- output이 sequence label, residue label, region label, function prediction, downstream interaction claim 중 무엇인가?

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/entities|Objects and entities]]
- [[molecular-modeling/proteins|Proteins]]
- [[molecular-modeling/genome|Genome]]
- [[ai/architectures|Architectures]]
- [[ai/learning-methods|Learning methods]]
