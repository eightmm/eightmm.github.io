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

where $s_{1:L}$ is a token sequence and $c$ is context such as organism, target family, region, assay, mutation, or annotation source.

## Route Map

| Question | Start | Watch |
| --- | --- | --- |
| What sequence object is modeled? | [Objects and Entities](/molecular-modeling/entities), [Sequence](/entities/sequence) | sequence may be a protein chain, genome window, variant context, or derived token stream |
| Is it protein sequence modeling? | [Proteins](/molecular-modeling/proteins), [Protein](/entities/protein) | homolog leakage, isoform choice, construct, mutation, truncation |
| Is it genome-level sequence modeling? | [Genome](/molecular-modeling/genome), [Genome](/entities/genome) | reference coordinate mismatch, duplicated windows, annotation leakage |
| Is the representation learned or fixed? | [Embedding](/concepts/architectures/embedding), [Tokenization](/concepts/architectures/tokenization) | pooling and token policy can change the claim |
| What split supports the claim? | [Protein family split](/concepts/evaluation/protein-family-split), [Leakage](/concepts/evaluation/leakage) | random row split can overstate generalization |

## Main Subroutes

| Area | Use For | Start |
| --- | --- | --- |
| Protein sequence | protein language models, mutation effect, family-aware representation, sequence-to-function | [Proteins](/molecular-modeling/proteins) |
| Genome sequence | genome window, k-mer, annotation, variant-effect prediction | [Genome](/molecular-modeling/genome) |
| Sequence representation | tokenization, embedding, pooling, sequence length, context window | [Sequence](/concepts/modalities/sequence) |
| Sequence evaluation | family split, near-duplicate control, label-source boundary | [Data and Evaluation](/molecular-modeling/data-evaluation) |

## Sequence vs Structure

Sequence-based modeling is not the opposite of structure-based modeling. Many workflows start from sequence and later use predicted or experimental structure:

$$
s_{1:L}
\rightarrow
h_{1:L}
\rightarrow
X
\rightarrow
\hat{y}
$$

Use this page when the primary input is sequence. Move to [[molecular-modeling/structure-based/index|Structure-based modeling]] when coordinates, pockets, poses, or complexes become first-class objects.

## Boundary

- Sequence-based: sequence token, mutation, region, family split, sequence representation이 claim의 중심입니다.
- Structure-based: coordinate, pocket, pose, conformer, complex geometry가 claim의 중심입니다.
- AI: Transformer, SSM, GNN, SSL 같은 architecture나 learning method 자체가 claim의 중심입니다.
- Math: likelihood, embedding similarity, gradient, geometry 식 자체를 설명합니다.

## Checks

- Is one example a protein chain, sequence window, variant-centered window, or annotated region?
- Are homologs, near duplicates, and repeated windows separated across train/test?
- Is the model using structure, templates, MSAs, or annotations that are unavailable at inference time?
- Is the output a sequence label, residue label, region label, function prediction, or downstream interaction claim?

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/entities|Objects and entities]]
- [[molecular-modeling/proteins|Proteins]]
- [[molecular-modeling/genome|Genome]]
- [[ai/architectures|Architectures]]
- [[ai/learning-methods|Learning methods]]
