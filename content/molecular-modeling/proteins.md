---
title: Proteins
aliases:
  - computational-biology/proteins
  - bio/proteins
tags:
  - computational-biology
  - proteins
---


# Proteins

Protein modeling은 sequence, structure, domain, binding site, learned representation을 다룹니다. 중요한 구분은 model이 sequence만 보는지, predicted structure를 보는지, experimental structure를 보는지, known complex를 보는지입니다.

This page owns the protein object layer: sequence, chain, construct, domain, structure source, and representation unit. Use [[molecular-modeling/protein-modeling|Protein Modeling]] for broader protein-modeling task maps, [[molecular-modeling/sequence-based|Sequence-Based Modeling]] for sequence-first routes, and [[molecular-modeling/structure-based/index|Structure-Based Modeling]] when pocket, pose, or complex geometry is central.

$$
r_P = \phi(s_{1:L}, X, c)
$$

여기서 $s_{1:L}$은 residue sequence, $X$는 optional coordinate information, $c$는 family, domain, pocket, mutation, assay condition 같은 context입니다.

## Route Map

| 질문 | 시작점 | 주의점 |
| --- | --- | --- |
| modeled object가 무엇인가? | [Protein](/entities/protein), [Sequence](/entities/sequence), [Structure](/entities/structure) | chain choice, isoform, construct, mutation, missing residue |
| input이 sequence-only인가 structure-aware인가? | [Protein representation](/concepts/protein-modeling/protein-representation), [Protein structure prediction](/concepts/protein-modeling/protein-structure-prediction) | predicted 또는 template-derived structure를 deployment에서 항상 가능한 정보처럼 쓰는 문제 |
| 어떤 biological unit을 보존하는가? | [Protein domain](/concepts/protein-modeling/protein-domain), [Sequence identity clustering](/concepts/protein-modeling/sequence-identity-clustering) | homolog leakage와 domain truncation |
| structure preprocessing이 method의 일부인가? | [Protein structure cleaning](/concepts/protein-modeling/protein-structure-cleaning), [Residue indexing](/concepts/protein-modeling/residue-indexing) | silent residue renumbering, missing atom, chain filtering |
| binding context가 task의 일부인가? | [Binding site](/concepts/protein-modeling/binding-site), [Pocket representation](/concepts/protein-modeling/pocket-representation), [Protein-ligand complex](/entities/protein-ligand-complex) | apo/holo distinction과 ligand-defined pocket |

## Representation Choices

| Representation | 쓰임 | 주요 Risk |
| --- | --- | --- |
| Raw sequence | language-model pretraining, classification, mutation effect prediction | homolog leakage와 truncation policy가 결과를 지배할 수 있음 |
| MSA / evolutionary profile | structure prediction, family-aware representation | MSA depth와 template/database overlap이 test information을 leak할 수 있음 |
| Residue embedding | downstream supervised model과 retrieval | pooling rule과 special-token handling이 representation을 바꿈 |
| Contact map / residue graph | full coordinate 없이 structure-aware prediction | threshold choice와 missing residue가 graph topology에 영향 |
| 3D coordinates | pocket, docking, structure refinement, equivariant model | coordinate source, chain selection, alignment, unit을 명시해야 함 |

## Sequence to Structure Map

Many protein notes move through this chain:

$$
s_{1:L}
\rightarrow
h_{1:L}
\rightarrow
G_P\ \text{or}\ X_P
\rightarrow
\hat{y}
$$

여기서 $s_{1:L}$은 amino-acid sequence, $h_{1:L}$은 residue-level representation, $G_P$는 residue/contact graph, $X_P$는 coordinate set, $\hat{y}$는 task output입니다.

## Sequence and Structure Routes

| Area | Start | 쓰임 |
| --- | --- | --- |
| Evolutionary context | [Multiple sequence alignment](/concepts/protein-modeling/multiple-sequence-alignment), [Sequence identity clustering](/concepts/protein-modeling/sequence-identity-clustering) | homolog control, family split, MSA-dependent methods |
| Structure graph | [Contact map](/concepts/protein-modeling/contact-map), [Sequence-structure alignment](/concepts/protein-modeling/sequence-structure-alignment) | residue graph construction and coordinate-aware representations |
| Binding context | [Binding site](/concepts/protein-modeling/binding-site), [Pocket representation](/concepts/protein-modeling/pocket-representation), [Pocket](/entities/pocket) | pocket-level prediction, docking, protein-ligand interaction |

## Claim Map

| Claim | 필요한 Boundary |
| --- | --- |
| Sequence representation works | sequence identity split, pooling rule, model-selection protocol |
| Structure representation helps | structure source, cleaning protocol, residue alignment, missing-region handling |
| Binding-site prediction works | pocket definition, ligand availability, apo/holo distinction, localization metric |
| Protein-ligand modeling generalizes | protein-family split plus ligand scaffold 또는 complex-pair split |

## Checks

- homolog와 protein family가 train/test 사이에서 분리되어 있는가?
- residue indexing, missing residue, mutation, chain choice가 explicit한가?
- structure source가 experimental, predicted, apo, holo, complex 중 무엇인가?
- model이 task를 바꾸는 template, MSA, bound ligand를 사용하는가?

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/interactions|Interaction modeling]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
