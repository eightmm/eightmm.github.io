---
title: Objects and Entities
aliases:
  - computational-biology/entities
  - bio/entities
tags:
  - computational-biology
  - entities
---


# Objects and Entities

Computational biology에서 먼저 정해야 하는 것은 모델이 다루는 대상입니다. 같은 단어라도 protein, ligand, target, assay, structure가 어떤 단위로 정의되는지에 따라 split, leakage, evaluation이 달라집니다.

$$
x_{\mathrm{bio}}
\in
\{\text{protein}, \text{ligand}, \text{pocket}, \text{complex}, \text{assay}, \text{genome region}\}
$$

## Core Objects

| Object | Use when | Main risk |
| --- | --- | --- |
| [Target](/entities/target) | target-conditioned activity, affinity, selectivity, assay context | target identity가 isoform, construct, mutation, species, assay source를 숨길 수 있음 |
| [Protein](/entities/protein) | sequence, structure, representation, family split | homolog leakage와 residue-index mismatch |
| [Pocket](/entities/pocket) | binding site, docking, pose prediction, structure-based generation | pocket이 ligand-defined이거나 inference에서 unavailable할 수 있음 |
| [Ligand](/entities/ligand) | target-bound molecule, pose, activity, interaction | ligand state와 atom mapping이 protocol마다 달라질 수 있음 |
| [Molecule](/entities/molecule) | property prediction, molecular generation, similarity, scaffold split | salt, stereo, tautomer, protonation, duplicate handling |
| [Protein-ligand complex](/entities/protein-ligand-complex) | docking, pose quality, interaction prediction, complex graph | pair-level split이 protein 또는 ligand family를 leak할 수 있음 |
| [Sequence](/entities/sequence) | protein language model, genome region model, motif task | sequence context와 split unit이 biological claim과 맞지 않을 수 있음 |
| [Structure](/entities/structure) | coordinate, conformer, protein structure, complex | template, frame, coordinate-source leakage |
| [Genome](/entities/genome) | sequence/region/variant-level modeling only | broad omics claim은 명시적으로 열기 전까지 scope 밖 |

여러 object와 relation이 함께 중요하면 [[entities/entity-relation-map|Entity relation map]]을 사용합니다.

## Entity Tuple

Paper note에서는 modeled unit을 단일 단어가 아니라 tuple로 적습니다.

$$
u
=
(\text{entity}, \text{context}, \text{measurement}, \text{split group})
$$

예를 들어 property prediction에서는 row가 molecule-only일 수 있지만, target-conditioned activity에는 molecule, target, assay, endpoint, unit, source가 필요합니다. Protein-ligand pose에는 protein structure, pocket rule, ligand state, atom mapping, pose coordinate, pose source가 필요합니다.

## Split Implication

| Claim | Hold out |
| --- | --- |
| new molecule generalization | scaffold, chemical series, time split |
| new protein generalization | sequence identity, family, fold, target split |
| new protein-ligand pair generalization | ligand/protein 양쪽 axis 또는 explicit pair-level claim |
| new structure generalization | template-aware, homolog-aware, coordinate-source-aware split |
| new assay generalization | assay/source split 또는 endpoint harmonization check |
| new genome region generalization | chromosome, locus, family, annotation-source split |

## Label Objects

- [[entities/target-assay-label|Target-assay-label contract]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]

## Checks

- entity가 biological object, chemical object, assay record, derived feature 중 무엇인가?
- isoform, construct, mutation, chain, ligand state, assay context가 explicit한가?
- one row가 object, pair, pose, measurement, generated sample, candidate list 중 무엇인가?
- label이 있을 때 target, assay, endpoint, unit, threshold, censoring, source가 보존되는가?
- representation이 inference에서 unavailable한 ligand-defined pocket, known pose, template, homolog, future annotation을 쓰는가?
- split unit이 biological claim과 같은가?
- input에 deployment time에 unavailable한 정보가 포함되는가?

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/sequence-based|Sequence-based modeling]]
- [[molecular-modeling/molecular-ligand|Molecular and ligand modeling]]
- [[molecular-modeling/interactions|Interaction modeling]]
- [[molecular-modeling/data-evaluation|Data and evaluation]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[concepts/modalities/representation-contract|Representation contract]]
- [[concepts/evaluation/leakage|Leakage]]
