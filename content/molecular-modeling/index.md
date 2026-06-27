---
title: Computational Biology
aliases:
  - computational-biology
  - comp-bio
  - bio
  - bio-ai
  - molecular-modeling
tags:
  - molecular-modeling
---


# Computational Biology

이 영역은 넓은 biology 전체가 아니라 molecule, protein, ligand, pocket, complex, conformer, docking, genome sequence처럼 계산 모델링에서 직접 다루는 객체와 workflow에 집중합니다. `Computational Biology`를 큰 이름으로 쓰고, 그 안에서 docking, conformer, structure-based modeling은 `Molecular Modeling` 하위 영역으로 봅니다.

AI 모델은 이 안의 방법 중 하나입니다. Docking이나 conformer generation 자체는 먼저 computational biology 또는 molecular modeling 문제로 보고, GNN, Transformer, diffusion, flow matching 같은 모델 구조와 학습법은 [[ai/index|AI]] 쪽에서 설명합니다.

반복해서 등장하는 기본 형태는 화학적/구조적 객체와 context를 모델 입력으로 두는 것입니다.

$$
\hat{y} = f_\theta(x_{\mathrm{mol}}, x_{\mathrm{context}})
$$

여기서 $x_{\mathrm{mol}}$은 molecule, protein sequence, structure, conformer, complex일 수 있고, $x_{\mathrm{context}}$는 pocket, target condition, assay context일 수 있습니다.

## 이름을 고르는 기준

| Name | Use When | Avoid When |
| --- | --- | --- |
| Computational Biology | protein, molecule, ligand, pocket, complex, structure, genome sequence가 함께 등장하는 큰 입구가 필요할 때 | broad omics, clinical biology, systems biology까지 여는 뜻으로 쓰는 경우 |
| Molecular Modeling | molecule, conformer, docking, pose, force field, virtual screening처럼 화학/구조 workflow가 중심일 때 | protein sequence-only나 genome sequence task까지 모두 부르는 경우 |
| Protein Modeling | protein sequence, structure, domain, binding site, design, interaction이 중심일 때 | ligand chemistry나 assay label이 더 중심인 경우 |
| Structure-Based Modeling | protein-ligand geometry, pocket, pose, interaction, scoring이 중심일 때 | 모델 architecture 자체가 주제인 경우 |
| AI | GNN, Transformer, diffusion, SSL, objective, architecture, training method가 중심일 때 | 생물학적 object, assay, split, coordinate protocol이 더 중요한 경우 |
| Math | likelihood, loss, gradient, geometry, symmetry, metric formula를 설명해야 할 때 | 실제 workflow나 model family를 설명하는 경우 |

## 먼저 볼 지도

| Area | Use For | Start |
| --- | --- | --- |
| Entities | protein, molecule, ligand, pocket, complex, assay, sequence, structure | [Entities](/molecular-modeling/entities) |
| Molecules | standardization, molecular graphs, fingerprints, conformers | [Molecules](/molecular-modeling/molecules) |
| Proteins | sequence, structure, domains, binding sites, representation | [Proteins](/molecular-modeling/proteins) |
| Structure-based modeling | protein-ligand geometry, interaction, scoring, generation | [Structure-based modeling](/molecular-modeling/structure-based) |
| Docking | preparation, pose generation, scoring, filtering, evaluation | [Docking](/molecular-modeling/docking) |
| Data and evaluation | label semantics, split units, leakage, assay harmonization | [Data and evaluation](/molecular-modeling/data-evaluation) |
| Geometry | coordinates, frames, invariance, equivariance | [Geometry](/molecular-modeling/geometry) |
| Genome | sequence, region, k-mer, variant-level modeling | [Genome](/molecular-modeling/genome) |

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
| Identity and cleanup | [Molecular standardization](/concepts/molecular-modeling/molecular-standardization), [Molecular identity](/concepts/molecular-modeling/molecular-identity), [Chemical state contract](/concepts/molecular-modeling/chemical-state-contract) |
| Representation | [Molecular featurization contract](/concepts/molecular-modeling/molecular-featurization-contract), [SMILES](/concepts/molecular-modeling/smiles), [Molecular graph](/concepts/molecular-modeling/molecular-graph), [Molecular fingerprint](/concepts/molecular-modeling/molecular-fingerprint) |
| Graph input | [Graph construction](/concepts/architectures/graph-construction), [Graph neural networks](/concepts/architectures/gnn), [Graph Transformer](/concepts/architectures/graph-transformer) |
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
| Chemical state | [Chemical state contract](/concepts/molecular-modeling/chemical-state-contract), [Molecular identity](/concepts/molecular-modeling/molecular-identity) |
| Evaluation protocol | [Evaluation protocol](/concepts/evaluation/evaluation-protocol), [Leakage](/concepts/evaluation/leakage) |
| Computational biology-specific splits | [Scaffold split](/concepts/evaluation/scaffold-split), [Protein family split](/concepts/evaluation/protein-family-split), [Protein-ligand split](/concepts/sbdd/protein-ligand-split) |
| Assay integration | [Assay harmonization](/concepts/evaluation/assay-harmonization) |
| Benchmark traps | [Negative set](/concepts/evaluation/negative-set), [Activity cliff](/concepts/evaluation/activity-cliff), [Applicability domain](/concepts/evaluation/applicability-domain), [Boltzmann ceiling analysis](/concepts/evaluation/boltzmann-ceiling) |

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

## 운영 문서

이 문서들은 주제 탐색용 사이드바에서는 숨기고, 논문 정리와 글 작성 단계에서 사용합니다.

| Need | Start |
| --- | --- |
| Scope and naming | [Computational Biology Scope](/molecular-modeling/modeling-scope) |
| Boundary and object claim | [Computational Biology Boundary](/molecular-modeling/computational-biology) |
| Paper intake | [Computational Biology paper intake](/molecular-modeling/paper-intake) |
| Paper claim patterns | [Computational Biology paper claim patterns](/molecular-modeling/paper-claim-patterns) |
| Cross-axis contract | [AI Computational Biology Math contract](/concepts/ai-computational-biology-math-contract) |
| Coverage check | [Coverage matrix](/concepts/coverage-matrix) |
| Korean synthesis post intake | [AI Computational Biology Math post intake](/posts/ai-molecular-math-post-intake) |

## Geometry, Structure, and Evaluation

| Topic | Start |
| --- | --- |
| Geometry basics | [Geometry](/molecular-modeling/geometry), [Geometry](/concepts/math/geometry), [Symmetry group](/concepts/math/symmetry-group) |
| Modalities and tasks | [Graph](/concepts/modalities/graph), [3D structure](/concepts/modalities/3d-structure), [Coordinate prediction](/concepts/tasks/coordinate-prediction), [Graph prediction](/concepts/tasks/graph-prediction) |
| Geometric DL | [Geometric deep learning](/concepts/geometric-deep-learning), [Equivariance](/concepts/geometric-deep-learning/equivariance), [Equivariant GNN](/concepts/geometric-deep-learning/equivariant-gnn) |
| Coordinates and features | [Coordinate modeling contract](/concepts/geometric-deep-learning/coordinate-modeling-contract), [Coordinate frame](/concepts/geometric-deep-learning/coordinate-frame), [Distance geometry](/concepts/geometric-deep-learning/distance-geometry), [Coordinate update](/concepts/geometric-deep-learning/coordinate-update), [Invariant feature](/concepts/geometric-deep-learning/invariant-feature), [Equivariant feature](/concepts/geometric-deep-learning/equivariant-feature) |
| Evaluation risk | [Evaluation](/concepts/evaluation), [Leakage](/concepts/evaluation/leakage) |

## Computational Biology 논문을 읽을 때

Computational biology 쪽 논문은 모델 성능보다 object, label, split, leakage를 먼저 분리해야 합니다.

| 먼저 볼 것 | 확인할 내용 | Start |
| --- | --- | --- |
| Modeled object | molecule, protein, ligand, pocket, complex, genome region 중 무엇인가 | [Entities](/molecular-modeling/entities) |
| Representation | string, graph, fingerprint, embedding, conformer, coordinate, complex graph 중 무엇인가 | [Molecules](/molecular-modeling/molecules), [Proteins](/molecular-modeling/proteins), [Geometry](/molecular-modeling/geometry) |
| Chemical state | salt, stereo, tautomer, protonation, charge, conformer policy가 명확한가 | [Chemical state contract](/concepts/molecular-modeling/chemical-state-contract) |
| Representation contract | raw object가 model-ready input으로 어떻게 바뀌는가 | [Representation contract](/concepts/modalities/representation-contract) |
| Label context | target, assay, endpoint, unit, threshold, censoring, source가 명확한가 | [Data and evaluation](/molecular-modeling/data-evaluation), [Target-assay-label contract](/entities/target-assay-label) |
| Structure context | apo/holo, predicted/experimental, pocket-defined/blind, ligand-defined 여부가 명확한가 | [Structure-based modeling](/molecular-modeling/structure-based), [Docking](/molecular-modeling/docking) |
| Split unit | scaffold, protein family, complex pair, assay/source, time 중 무엇으로 나누는가 | [Data and evaluation](/molecular-modeling/data-evaluation) |
| Benchmark trap | negative set, activity cliff, applicability domain, assay harmonization 문제가 있는가 | [Data and evaluation](/molecular-modeling/data-evaluation), [Coverage matrix](/concepts/coverage-matrix) |
| Evaluation claim | pose, affinity, ranking, enrichment, property, generation 중 무엇을 주장하는가 | [Docking](/molecular-modeling/docking), [Evaluation](/ai/evaluation) |
| Public boundary | 내부 데이터나 미공개 결과 없이 일반화 가능한가 | [Computational Biology Boundary](/molecular-modeling/computational-biology) |
| Intake protocol | 위 항목들을 한 번에 점검할 paper note인가 | [Computational Biology paper intake](/molecular-modeling/paper-intake) |
| Claim pattern | property, activity, docking, generation, protein design, genome sequence 중 어떤 형태인가 | [Computational Biology paper claim patterns](/molecular-modeling/paper-claim-patterns) |

## 관련 입구

- [[ai/index|AI]]
- [[papers/index|Papers]]
- [[projects/index|Projects]]
- [[molecular-modeling/paper-intake|Computational Biology paper intake]]
- [[molecular-modeling/paper-claim-patterns|Computational Biology paper claim patterns]]
- [[molecular-modeling/modeling-scope|Computational Biology scope]]
- [[concepts/ai-computational-biology-math-contract|AI Computational Biology Math contract]]
- [[concepts/coverage-matrix|Coverage matrix]]
