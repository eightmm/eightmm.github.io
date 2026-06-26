---
title: Molecular Modeling
aliases:
  - bio
  - bio-ai
  - molecular-modeling
tags:
  - bio
  - molecular-modeling
---


# Molecular Modeling

이 영역은 넓은 biology 전체가 아니라 molecule, protein, ligand, pocket, complex, conformer, docking처럼 계산 모델링에서 직접 다루는 객체와 workflow에 집중합니다. AI 모델은 이 안의 방법 중 하나이고, docking이나 conformer generation 자체는 먼저 molecular modeling 문제로 봅니다.

반복해서 등장하는 기본 형태는 화학적/구조적 객체와 context를 모델 입력으로 두는 것입니다.

$$
\hat{y} = f_\theta(x_{\mathrm{mol}}, x_{\mathrm{context}})
$$

여기서 $x_{\mathrm{mol}}$은 molecule, protein sequence, structure, conformer, complex일 수 있고, $x_{\mathrm{context}}$는 pocket, target condition, assay context일 수 있습니다.

## 먼저 볼 지도

| Area | Use For | Start |
| --- | --- | --- |
| Scope and naming | what belongs under Molecular Modeling vs AI, Math, Agents | [Molecular Modeling Scope](/molecular-modeling/modeling-scope) |
| Scope and objects | object, measurement, representation, claim boundary | [Computational Biology](/molecular-modeling/computational-biology) |
| Entities | protein, molecule, ligand, pocket, complex, assay, sequence, structure | [Entities](/molecular-modeling/entities) |
| Molecules | standardization, molecular graphs, fingerprints, conformers | [Molecules](/molecular-modeling/molecules) |
| Proteins | sequence, structure, domains, binding sites, representation | [Proteins](/molecular-modeling/proteins) |
| Structure-based modeling | protein-ligand geometry, interaction, scoring, generation | [Structure-based modeling](/molecular-modeling/structure-based) |
| Docking | preparation, pose generation, scoring, filtering, evaluation | [Docking](/molecular-modeling/docking) |
| Data and evaluation | label semantics, split units, leakage, assay harmonization | [Data and evaluation](/molecular-modeling/data-evaluation) |
| Geometry | coordinates, frames, invariance, equivariance | [Geometry](/molecular-modeling/geometry) |
| Genome | sequence, region, k-mer, variant-level modeling | [Genome](/molecular-modeling/genome) |
| Paper intake | object, representation, label, split, leakage, public boundary | [Molecular modeling paper intake](/molecular-modeling/paper-intake) |
| Post intake | AI, molecular modeling, and math synthesis writing | [AI-Molecular-Math post intake](/posts/ai-bio-math-post-intake) |
| Coverage matrix | object, data, model, objective, evidence, public boundary | [Coverage matrix](/concepts/coverage-matrix) |

## 다루는 객체

| Object | Start | Why It Matters |
| --- | --- | --- |
| Entity map | [Entities](/molecular-modeling/entities), [Entity relation map](/entities/entity-relation-map) | 전체 객체 관계를 먼저 잡습니다. |
| Protein / target | [Protein](/entities/protein), [Target](/entities/target) | sequence, structure, family split, target context의 기준입니다. |
| Molecule / ligand | [Molecule](/entities/molecule), [Ligand](/entities/ligand) | standardization, scaffold, property, pose의 기준입니다. |
| Pocket / complex | [Pocket](/entities/pocket), [Protein-ligand complex](/entities/protein-ligand-complex) | interaction, docking, pose quality를 정의합니다. |
| Sequence / structure | [Sequence](/entities/sequence), [Structure](/entities/structure) | sequence model과 coordinate model을 연결합니다. |
| Label context | [Target-assay-label contract](/entities/target-assay-label), [Bioactivity label](/entities/bioactivity-label) | assay, endpoint, unit, censoring을 분리합니다. |
| Genome | [Genome](/entities/genome) | broad omics가 아니라 sequence/region/variant 입력으로 다룹니다. |

## Molecule and Ligand Modeling

| Topic | Start |
| --- | --- |
| Overview | [Molecules](/molecular-modeling/molecules), [Molecular modeling concepts](/concepts/molecular-modeling) |
| Identity and cleanup | [Molecular standardization](/concepts/molecular-modeling/molecular-standardization), [Molecular identity](/concepts/molecular-modeling/molecular-identity) |
| Representation | [Molecular featurization contract](/concepts/molecular-modeling/molecular-featurization-contract), [SMILES](/concepts/molecular-modeling/smiles), [Molecular graph](/concepts/molecular-modeling/molecular-graph), [Molecular fingerprint](/concepts/molecular-modeling/molecular-fingerprint) |
| Tasks | [Property prediction](/concepts/tasks/property-prediction), [Molecular property prediction](/concepts/molecular-modeling/molecular-property-prediction), [Molecular similarity](/concepts/molecular-modeling/molecular-similarity) |
| Structure and chemistry details | [Conformer](/concepts/molecular-modeling/conformer), [Tautomer](/concepts/molecular-modeling/tautomer), [Protonation state](/concepts/molecular-modeling/protonation-state), [Stereochemistry](/concepts/molecular-modeling/stereochemistry) |
| Geometry protocols | [Conformer](/concepts/molecular-modeling/conformer), [Force field](/concepts/molecular-modeling/force-field), [Energy minimization](/concepts/molecular-modeling/energy-minimization), [Molecular dynamics](/concepts/molecular-modeling/molecular-dynamics) |
| Conformer-dependent modeling | [Conformer](/concepts/molecular-modeling/conformer), [Coordinate modeling contract](/concepts/geometric-deep-learning/coordinate-modeling-contract) |
| Search | [Substructure search](/concepts/molecular-modeling/substructure-search) |

## Structure-Based Modeling

| Topic | Start |
| --- | --- |
| Overview | [Structure-based modeling](/molecular-modeling/structure-based), [SBDD concepts](/concepts/sbdd) |
| Core tasks | [Interaction prediction](/concepts/tasks/interaction-prediction), [Localization](/concepts/tasks/localization), [Coordinate prediction](/concepts/tasks/coordinate-prediction), [Graph prediction](/concepts/tasks/graph-prediction) |
| Docking workflow | [Docking](/molecular-modeling/docking), [Docking workflow](/concepts/sbdd/docking-workflow), [Receptor and ligand preparation](/concepts/sbdd/receptor-ligand-preparation) |
| Pose and interaction | [Pose generation](/concepts/sbdd/pose-generation), [Pose RMSD](/concepts/sbdd/pose-rmsd), [Pose quality](/concepts/sbdd/pose-quality), [Protein-ligand interaction](/concepts/sbdd/protein-ligand-interaction) |
| Scoring and screening | [Scoring function](/concepts/sbdd/scoring-function), [Binding affinity](/concepts/sbdd/binding-affinity), [Virtual screening](/concepts/sbdd/virtual-screening), [Interaction fingerprint](/concepts/sbdd/interaction-fingerprint) |
| Classical geometry support | [Force field](/concepts/molecular-modeling/force-field), [Energy minimization](/concepts/molecular-modeling/energy-minimization), [Molecular dynamics](/concepts/molecular-modeling/molecular-dynamics) |
| Benchmark risk | [Template leakage](/concepts/sbdd/template-leakage), [PoseBusters](/papers/sbdd/posebusters) |

## Data, Labels, and Splits

| Concern | Start |
| --- | --- |
| Data construction | [Data and evaluation](/molecular-modeling/data-evaluation), [Dataset construction checklist](/concepts/data/dataset-construction-checklist) |
| Example and split units | [Example unit](/concepts/data/example-unit), [Split unit](/concepts/data/split-unit) |
| Label semantics | [Label semantics](/concepts/data/label-semantics), [Target-assay-label contract](/entities/target-assay-label), [Bioactivity label](/entities/bioactivity-label) |
| Preprocessing | [Preprocessing contract](/concepts/data/preprocessing-contract) |
| Evaluation protocol | [Evaluation protocol](/concepts/evaluation/evaluation-protocol), [Leakage](/concepts/evaluation/leakage) |
| Molecular Modeling-specific splits | [Scaffold split](/concepts/evaluation/scaffold-split), [Protein family split](/concepts/evaluation/protein-family-split), [Protein-ligand split](/concepts/sbdd/protein-ligand-split) |
| Assay integration | [Assay harmonization](/concepts/evaluation/assay-harmonization) |

## Protein and Sequence Modeling

| Topic | Start |
| --- | --- |
| Overview | [Proteins](/molecular-modeling/proteins), [Protein modeling concepts](/concepts/protein-modeling) |
| Representation | [Protein representation](/concepts/protein-modeling/protein-representation), [Self-supervised learning](/concepts/learning/self-supervised-learning) |
| Structure | [Protein structure cleaning](/concepts/protein-modeling/protein-structure-cleaning), [Protein structure prediction](/concepts/protein-modeling/protein-structure-prediction), [Contact map](/concepts/protein-modeling/contact-map) |
| Domain and site | [Protein domain](/concepts/protein-modeling/protein-domain), [Binding site](/concepts/protein-modeling/binding-site), [Pocket representation](/concepts/protein-modeling/pocket-representation) |
| Sequence handling | [Multiple sequence alignment](/concepts/protein-modeling/multiple-sequence-alignment), [Sequence identity clustering](/concepts/protein-modeling/sequence-identity-clustering), [Residue indexing](/concepts/protein-modeling/residue-indexing) |

## Genome-Level Sequence Modeling

유전체는 이 블로그에서 넓은 omics 주제가 아니라 sequence/region/variant 수준의 입력 객체로만 다룹니다.

- [[molecular-modeling/genome|Genome]]
- [[concepts/genome-modeling/index|Genome modeling concepts]]
- [[concepts/genome-modeling/genomic-region|Genomic region]]
- [[concepts/genome-modeling/k-mer|K-mer]]
- [[concepts/genome-modeling/variant-effect-prediction|Variant effect prediction]]
- [[concepts/genome-modeling/genome-annotation|Genome annotation]]

## Geometry, Structure, and Evaluation

| Topic | Start |
| --- | --- |
| Geometry basics | [Geometry](/molecular-modeling/geometry), [Geometry](/concepts/math/geometry), [Symmetry group](/concepts/math/symmetry-group) |
| Modalities and tasks | [Graph](/concepts/modalities/graph), [3D structure](/concepts/modalities/3d-structure), [Coordinate prediction](/concepts/tasks/coordinate-prediction), [Graph prediction](/concepts/tasks/graph-prediction) |
| Geometric DL | [Geometric deep learning](/concepts/geometric-deep-learning), [Equivariance](/concepts/geometric-deep-learning/equivariance), [Equivariant GNN](/concepts/geometric-deep-learning/equivariant-gnn) |
| Coordinates and features | [Coordinate frame](/concepts/geometric-deep-learning/coordinate-frame), [Distance geometry](/concepts/geometric-deep-learning/distance-geometry), [Invariant feature](/concepts/geometric-deep-learning/invariant-feature), [Equivariant feature](/concepts/geometric-deep-learning/equivariant-feature) |
| Evaluation risk | [Evaluation](/concepts/evaluation), [Leakage](/concepts/evaluation/leakage) |

## Molecular Modeling 논문을 읽을 때

Molecular modeling 쪽 논문은 모델 성능보다 object, label, split, leakage를 먼저 분리해야 합니다.

| 먼저 볼 것 | 확인할 내용 | Start |
| --- | --- | --- |
| Modeled object | molecule, protein, ligand, pocket, complex, genome region 중 무엇인가 | [Entities](/molecular-modeling/entities) |
| Representation | string, graph, fingerprint, embedding, conformer, coordinate, complex graph 중 무엇인가 | [Molecules](/molecular-modeling/molecules), [Proteins](/molecular-modeling/proteins), [Geometry](/molecular-modeling/geometry) |
| Representation contract | raw object가 model-ready input으로 어떻게 바뀌는가 | [Representation contract](/concepts/modalities/representation-contract) |
| Label context | target, assay, endpoint, unit, threshold, censoring, source가 명확한가 | [Data and evaluation](/molecular-modeling/data-evaluation), [Target-assay-label contract](/entities/target-assay-label) |
| Structure context | apo/holo, predicted/experimental, pocket-defined/blind, ligand-defined 여부가 명확한가 | [Structure-based modeling](/molecular-modeling/structure-based), [Docking](/molecular-modeling/docking) |
| Split unit | scaffold, protein family, complex pair, assay/source, time 중 무엇으로 나누는가 | [Data and evaluation](/molecular-modeling/data-evaluation) |
| Evaluation claim | pose, affinity, ranking, enrichment, property, generation 중 무엇을 주장하는가 | [Docking](/molecular-modeling/docking), [Evaluation](/ai/evaluation) |
| Public boundary | 내부 데이터나 미공개 결과 없이 일반화 가능한가 | [Computational Biology](/molecular-modeling/computational-biology) |
| Intake protocol | 위 항목들을 한 번에 점검할 paper note인가 | [Molecular modeling paper intake](/molecular-modeling/paper-intake) |

## 관련 입구

- [[ai/index|AI]]
- [[papers/index|Papers]]
- [[projects/index|Projects]]
- [[molecular-modeling/paper-intake|Molecular modeling paper intake]]
- [[molecular-modeling/modeling-scope|Molecular modeling scope]]
- [[concepts/coverage-matrix|Coverage matrix]]
