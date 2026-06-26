---
title: Bio
aliases:
  - bio-ai
tags:
  - bio
---


# Bio

Bio 영역은 computational biology 전체를 넓게 훑기보다, AI와 직접 연결되는 구조 기반 모델링, 단백질, 분자, 리간드, docking, protein-ligand interaction, sequence-level genome modeling에 범위를 좁힙니다.

반복해서 등장하는 기본 형태는 생물학적/화학적 객체와 context를 모델 입력으로 두는 것입니다.

$$
\hat{y} = f_\theta(x_{\mathrm{bio}}, x_{\mathrm{context}})
$$

여기서 $x_{\mathrm{bio}}$는 sequence, molecule, structure, complex일 수 있고, $x_{\mathrm{context}}$는 pocket, target condition, assay context일 수 있습니다.

## 먼저 볼 지도

| Area | Use For | Start |
| --- | --- | --- |
| Computational biology | object, measurement, representation, claim boundary | [Computational Biology](/bio/computational-biology) |
| Entities | protein, molecule, ligand, pocket, complex, assay, sequence, structure | [Entities](/bio/entities) |
| Molecules | standardization, molecular graphs, fingerprints, conformers | [Molecules](/bio/molecules) |
| Proteins | sequence, structure, domains, binding sites, representation | [Proteins](/bio/proteins) |
| Structure-based AI | protein-ligand geometry, interaction, scoring, generation | [Structure-based AI](/bio/structure-based-ai) |
| Docking | preparation, pose generation, scoring, filtering, evaluation | [Docking](/bio/docking) |
| Data and evaluation | label semantics, split units, leakage, assay harmonization | [Data and evaluation](/bio/data-evaluation) |
| Geometry | coordinates, frames, invariance, equivariance | [Geometry](/bio/geometry) |
| Genome | sequence, region, k-mer, variant-level modeling | [Genome](/bio/genome) |

## 다루는 객체

| Object | Start | Why It Matters |
| --- | --- | --- |
| Entity map | [Entities](/bio/entities), [Entity relation map](/entities/entity-relation-map) | 전체 객체 관계를 먼저 잡습니다. |
| Protein / target | [Protein](/entities/protein), [Target](/entities/target) | sequence, structure, family split, target context의 기준입니다. |
| Molecule / ligand | [Molecule](/entities/molecule), [Ligand](/entities/ligand) | standardization, scaffold, property, pose의 기준입니다. |
| Pocket / complex | [Pocket](/entities/pocket), [Protein-ligand complex](/entities/protein-ligand-complex) | interaction, docking, pose quality를 정의합니다. |
| Sequence / structure | [Sequence](/entities/sequence), [Structure](/entities/structure) | sequence model과 coordinate model을 연결합니다. |
| Label context | [Target-assay-label contract](/entities/target-assay-label), [Bioactivity label](/entities/bioactivity-label) | assay, endpoint, unit, censoring을 분리합니다. |
| Genome | [Genome](/entities/genome) | broad omics가 아니라 sequence/region/variant 입력으로 다룹니다. |

## Molecule and Ligand Modeling

| Topic | Start |
| --- | --- |
| Overview | [Molecules](/bio/molecules), [Molecular modeling concepts](/concepts/molecular-modeling) |
| Identity and cleanup | [Molecular standardization](/concepts/molecular-modeling/molecular-standardization), [Molecular identity](/concepts/molecular-modeling/molecular-identity) |
| Representation | [Molecular featurization contract](/concepts/molecular-modeling/molecular-featurization-contract), [SMILES](/concepts/molecular-modeling/smiles), [Molecular graph](/concepts/molecular-modeling/molecular-graph), [Molecular fingerprint](/concepts/molecular-modeling/molecular-fingerprint) |
| Tasks | [Property prediction](/concepts/tasks/property-prediction), [Molecular property prediction](/concepts/molecular-modeling/molecular-property-prediction), [Molecular similarity](/concepts/molecular-modeling/molecular-similarity) |
| Structure and chemistry details | [Conformer](/concepts/molecular-modeling/conformer), [Tautomer](/concepts/molecular-modeling/tautomer), [Protonation state](/concepts/molecular-modeling/protonation-state), [Stereochemistry](/concepts/molecular-modeling/stereochemistry) |
| Search | [Substructure search](/concepts/molecular-modeling/substructure-search) |

## Structure-Based AI

| Topic | Start |
| --- | --- |
| Overview | [Structure-based AI](/bio/structure-based-ai), [SBDD concepts](/concepts/sbdd) |
| Core tasks | [Interaction prediction](/concepts/tasks/interaction-prediction), [Localization](/concepts/tasks/localization), [Coordinate prediction](/concepts/tasks/coordinate-prediction), [Graph prediction](/concepts/tasks/graph-prediction) |
| Docking workflow | [Docking](/bio/docking), [Docking workflow](/concepts/sbdd/docking-workflow), [Receptor and ligand preparation](/concepts/sbdd/receptor-ligand-preparation) |
| Pose and interaction | [Pose generation](/concepts/sbdd/pose-generation), [Pose RMSD](/concepts/sbdd/pose-rmsd), [Pose quality](/concepts/sbdd/pose-quality), [Protein-ligand interaction](/concepts/sbdd/protein-ligand-interaction) |
| Scoring and screening | [Scoring function](/concepts/sbdd/scoring-function), [Binding affinity](/concepts/sbdd/binding-affinity), [Virtual screening](/concepts/sbdd/virtual-screening), [Interaction fingerprint](/concepts/sbdd/interaction-fingerprint) |
| Benchmark risk | [Template leakage](/concepts/sbdd/template-leakage), [PoseBusters](/papers/sbdd/posebusters) |

## Data, Labels, and Splits

| Concern | Start |
| --- | --- |
| Data construction | [Data and evaluation](/bio/data-evaluation), [Dataset construction checklist](/concepts/data/dataset-construction-checklist) |
| Example and split units | [Example unit](/concepts/data/example-unit), [Split unit](/concepts/data/split-unit) |
| Label semantics | [Label semantics](/concepts/data/label-semantics), [Target-assay-label contract](/entities/target-assay-label), [Bioactivity label](/entities/bioactivity-label) |
| Preprocessing | [Preprocessing contract](/concepts/data/preprocessing-contract) |
| Evaluation protocol | [Evaluation protocol](/concepts/evaluation/evaluation-protocol), [Leakage](/concepts/evaluation/leakage) |
| Bio-specific splits | [Scaffold split](/concepts/evaluation/scaffold-split), [Protein family split](/concepts/evaluation/protein-family-split), [Protein-ligand split](/concepts/sbdd/protein-ligand-split) |
| Assay integration | [Assay harmonization](/concepts/evaluation/assay-harmonization) |

## Protein and Sequence Modeling

| Topic | Start |
| --- | --- |
| Overview | [Proteins](/bio/proteins), [Protein modeling concepts](/concepts/protein-modeling) |
| Representation | [Protein representation](/concepts/protein-modeling/protein-representation), [Self-supervised learning](/concepts/learning/self-supervised-learning) |
| Structure | [Protein structure cleaning](/concepts/protein-modeling/protein-structure-cleaning), [Protein structure prediction](/concepts/protein-modeling/protein-structure-prediction), [Contact map](/concepts/protein-modeling/contact-map) |
| Domain and site | [Protein domain](/concepts/protein-modeling/protein-domain), [Binding site](/concepts/protein-modeling/binding-site), [Pocket representation](/concepts/protein-modeling/pocket-representation) |
| Sequence handling | [Multiple sequence alignment](/concepts/protein-modeling/multiple-sequence-alignment), [Sequence identity clustering](/concepts/protein-modeling/sequence-identity-clustering), [Residue indexing](/concepts/protein-modeling/residue-indexing) |

## Genome-Level Sequence Modeling

유전체는 이 블로그에서 넓은 omics 주제가 아니라 sequence/region/variant 수준의 입력 객체로만 다룹니다.

- [[bio/genome|Genome]]
- [[concepts/genome-modeling/index|Genome modeling concepts]]
- [[concepts/genome-modeling/genomic-region|Genomic region]]
- [[concepts/genome-modeling/k-mer|K-mer]]
- [[concepts/genome-modeling/variant-effect-prediction|Variant effect prediction]]
- [[concepts/genome-modeling/genome-annotation|Genome annotation]]

## Geometry, Structure, and Evaluation

| Topic | Start |
| --- | --- |
| Geometry basics | [Geometry](/bio/geometry), [Geometry](/concepts/math/geometry), [Symmetry group](/concepts/math/symmetry-group) |
| Modalities and tasks | [Graph](/concepts/modalities/graph), [3D structure](/concepts/modalities/3d-structure), [Coordinate prediction](/concepts/tasks/coordinate-prediction), [Graph prediction](/concepts/tasks/graph-prediction) |
| Geometric DL | [Geometric deep learning](/concepts/geometric-deep-learning), [Equivariance](/concepts/geometric-deep-learning/equivariance), [Equivariant GNN](/concepts/geometric-deep-learning/equivariant-gnn) |
| Coordinates and features | [Coordinate frame](/concepts/geometric-deep-learning/coordinate-frame), [Distance geometry](/concepts/geometric-deep-learning/distance-geometry), [Invariant feature](/concepts/geometric-deep-learning/invariant-feature), [Equivariant feature](/concepts/geometric-deep-learning/equivariant-feature) |
| Evaluation risk | [Evaluation](/concepts/evaluation), [Leakage](/concepts/evaluation/leakage) |

## 관련 입구

- [[ai/index|AI]]
- [[papers/index|Papers]]
- [[projects/index|Projects]]
