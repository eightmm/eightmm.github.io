---
title: Bio-AI
tags:
  - bio-ai
---

# Bio-AI

Bio-AI 영역을 구조 기반 모델링, 단백질, 분자, 리간드, protein-ligand interaction 중심으로 정리하는 입구입니다. 전체 computational biology를 다루기보다, 실제로 다룰 가능성이 높은 구조 기반 AI와 단백질/분자 모델링에 범위를 좁힙니다.

반복해서 등장하는 기본 형태는 생물학적/화학적 객체와 context를 모델 입력으로 두는 것입니다.

$$
\hat{y} = f_\theta(x_{\mathrm{bio}}, x_{\mathrm{context}})
$$

여기서 $x_{\mathrm{bio}}$는 sequence, molecule, structure, complex일 수 있고, $x_{\mathrm{context}}$는 pocket, target condition, assay context일 수 있습니다.

## 다루는 객체

| Object | Start | Why It Matters |
| --- | --- | --- |
| Entity map | [[bio-ai/entities|Entities]], [[entities/entity-relation-map|Entity relation map]] | 전체 객체 관계를 먼저 잡습니다. |
| Protein / target | [[entities/protein|Protein]], [[entities/target|Target]] | sequence, structure, family split, target context의 기준입니다. |
| Molecule / ligand | [[entities/molecule|Molecule]], [[entities/ligand|Ligand]] | standardization, scaffold, property, pose의 기준입니다. |
| Pocket / complex | [[entities/pocket|Pocket]], [[entities/protein-ligand-complex|Protein-ligand complex]] | interaction, docking, pose quality를 정의합니다. |
| Sequence / structure | [[entities/sequence|Sequence]], [[entities/structure|Structure]] | sequence model과 coordinate model을 연결합니다. |
| Label context | [[entities/target-assay-label|Target-assay-label contract]], [[entities/bioactivity-label|Bioactivity label]] | assay, endpoint, unit, censoring을 분리합니다. |
| Genome | [[entities/genome|Genome]] | broad omics가 아니라 sequence/region/variant 입력으로 다룹니다. |

## Molecule and Ligand Modeling

| Topic | Start |
| --- | --- |
| Overview | [[bio-ai/molecules|Molecules]], [[concepts/molecular-modeling/index|Molecular modeling concepts]] |
| Identity and cleanup | [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]], [[concepts/molecular-modeling/molecular-identity|Molecular identity]] |
| Representation | [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]], [[concepts/molecular-modeling/smiles|SMILES]], [[concepts/molecular-modeling/molecular-graph|Molecular graph]], [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]] |
| Tasks | [[concepts/tasks/property-prediction|Property prediction]], [[concepts/molecular-modeling/molecular-property-prediction|Molecular property prediction]], [[concepts/molecular-modeling/molecular-similarity|Molecular similarity]] |
| Structure and chemistry details | [[concepts/molecular-modeling/conformer|Conformer]], [[concepts/molecular-modeling/tautomer|Tautomer]], [[concepts/molecular-modeling/protonation-state|Protonation state]], [[concepts/molecular-modeling/stereochemistry|Stereochemistry]] |
| Search | [[concepts/molecular-modeling/substructure-search|Substructure search]] |

## Structure-Based AI

| Topic | Start |
| --- | --- |
| Overview | [[bio-ai/structure-based-ai|Structure-based AI]], [[research/structure-based-ai/index|Structure-based AI]], [[concepts/sbdd/index|SBDD concepts]] |
| Core tasks | [[concepts/tasks/interaction-prediction|Interaction prediction]], [[concepts/tasks/localization|Localization]], [[concepts/tasks/coordinate-prediction|Coordinate prediction]], [[concepts/tasks/graph-prediction|Graph prediction]] |
| Docking workflow | [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]], [[concepts/sbdd/docking-workflow|Docking workflow]], [[concepts/sbdd/receptor-ligand-preparation|Receptor and ligand preparation]] |
| Pose and interaction | [[concepts/sbdd/pose-generation|Pose generation]], [[concepts/sbdd/pose-rmsd|Pose RMSD]], [[concepts/sbdd/pose-quality|Pose quality]], [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]] |
| Scoring and screening | [[concepts/sbdd/scoring-function|Scoring function]], [[concepts/sbdd/binding-affinity|Binding affinity]], [[concepts/sbdd/virtual-screening|Virtual screening]], [[concepts/sbdd/interaction-fingerprint|Interaction fingerprint]] |
| Benchmark risk | [[concepts/sbdd/template-leakage|Template leakage]], [[papers/sbdd/posebusters|PoseBusters]] |

## Data, Labels, and Splits

| Concern | Start |
| --- | --- |
| Data construction | [[bio-ai/data-evaluation|Data and evaluation]], [[concepts/data/dataset-construction-checklist|Dataset construction checklist]] |
| Example and split units | [[concepts/data/example-unit|Example unit]], [[concepts/data/split-unit|Split unit]] |
| Label semantics | [[concepts/data/label-semantics|Label semantics]], [[entities/target-assay-label|Target-assay-label contract]], [[entities/bioactivity-label|Bioactivity label]] |
| Preprocessing | [[concepts/data/preprocessing-contract|Preprocessing contract]] |
| Evaluation protocol | [[concepts/evaluation/evaluation-protocol|Evaluation protocol]], [[concepts/evaluation/leakage|Leakage]] |
| Bio-specific splits | [[concepts/evaluation/scaffold-split|Scaffold split]], [[concepts/evaluation/protein-family-split|Protein family split]], [[concepts/sbdd/protein-ligand-split|Protein-ligand split]] |
| Assay integration | [[concepts/evaluation/assay-harmonization|Assay harmonization]] |

## Protein and Sequence Modeling

| Topic | Start |
| --- | --- |
| Overview | [[bio-ai/proteins|Proteins]], [[research/protein-modeling/index|Protein modeling]], [[concepts/protein-modeling/index|Protein modeling concepts]] |
| Representation | [[concepts/protein-modeling/protein-representation|Protein representation]], [[concepts/learning/self-supervised-learning|Self-supervised learning]] |
| Structure | [[concepts/protein-modeling/protein-structure-cleaning|Protein structure cleaning]], [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]], [[concepts/protein-modeling/contact-map|Contact map]] |
| Domain and site | [[concepts/protein-modeling/protein-domain|Protein domain]], [[concepts/protein-modeling/binding-site|Binding site]], [[concepts/protein-modeling/pocket-representation|Pocket representation]] |
| Sequence handling | [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]], [[concepts/protein-modeling/sequence-identity-clustering|Sequence identity clustering]], [[concepts/protein-modeling/residue-indexing|Residue indexing]] |

## Genome-Level Sequence Modeling

유전체는 이 블로그에서 넓은 omics 주제가 아니라 sequence/region/variant 수준의 입력 객체로만 다룹니다.

- [[bio-ai/genome|Genome]]
- [[concepts/genome-modeling/index|Genome modeling concepts]]
- [[concepts/genome-modeling/genomic-region|Genomic region]]
- [[concepts/genome-modeling/k-mer|K-mer]]
- [[concepts/genome-modeling/variant-effect-prediction|Variant effect prediction]]
- [[concepts/genome-modeling/genome-annotation|Genome annotation]]

## Geometry, Structure, and Evaluation

| Topic | Start |
| --- | --- |
| Geometry basics | [[bio-ai/geometry|Geometry]], [[concepts/math/geometry|Geometry]], [[concepts/math/symmetry-group|Symmetry group]] |
| Modalities and tasks | [[concepts/modalities/graph|Graph]], [[concepts/modalities/3d-structure|3D structure]], [[concepts/tasks/coordinate-prediction|Coordinate prediction]], [[concepts/tasks/graph-prediction|Graph prediction]] |
| Geometric DL | [[concepts/geometric-deep-learning/index|Geometric deep learning]], [[concepts/geometric-deep-learning/equivariance|Equivariance]], [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]] |
| Coordinates and features | [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]], [[concepts/geometric-deep-learning/distance-geometry|Distance geometry]], [[concepts/geometric-deep-learning/invariant-feature|Invariant feature]], [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]] |
| Evaluation risk | [[concepts/evaluation/index|Evaluation]], [[concepts/evaluation/leakage|Leakage]] |

## 관련 입구

- [[ai/index|AI]]
- [[papers/index|Papers]]
- [[projects/index|Projects]]
