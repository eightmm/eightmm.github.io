---
title: Computational Biology
aliases:
  - computational-biology
  - comp-bio
  - bio
  - molecular-modeling
tags:
  - computational-biology
---

# Computational Biology

이 영역은 넓은 biology 전체가 아니라 계산 모델링에서 직접 다루는 객체와 workflow에 집중합니다. 주요 대상은 molecule, ligand, protein, pocket, protein-ligand complex, conformer, structure, genome sequence입니다. Transcriptomics, single-cell, clinical omics, systems biology처럼 범위가 급격히 넓어지는 주제는 실제 연구나 프로젝트 필요가 생길 때 별도로 엽니다.

AI 모델은 이 영역의 방법 중 하나입니다. 객체와 평가 조건은 여기서 정하고, GNN, Transformer, diffusion, flow matching, SSL 같은 모델 구조와 학습법은 [[ai/index|AI]]에서 봅니다.

$$
\hat{y}=f_\theta(x_{\mathrm{object}}, x_{\mathrm{context}})
$$

여기서 $x_{\mathrm{object}}$는 molecule, protein sequence, structure, conformer, complex, genome region일 수 있고, $x_{\mathrm{context}}$는 pocket, target, assay, split, coordinate source 같은 조건입니다.

## 먼저 볼 지도

| Area | Use For | Start |
| --- | --- | --- |
| Entities | protein, molecule, ligand, pocket, complex, assay, sequence, structure의 단위 정의 | [Entities](/molecular-modeling/entities) |
| Molecules | molecular identity, standardization, graph, fingerprint, conformer, property prediction | [Molecules](/molecular-modeling/molecules) |
| Proteins | sequence, structure, domain, binding site, representation, protein-family split | [Proteins](/molecular-modeling/proteins) |
| Structure-Based Modeling | pocket, protein-ligand geometry, docking, pose, scoring, virtual screening | [Structure-Based Modeling](/molecular-modeling/structure-based) |
| Data and Evaluation | label semantics, split unit, leakage, assay harmonization, benchmark traps | [Data and Evaluation](/molecular-modeling/data-evaluation) |
| Genome-Level Sequence | genome region, k-mer, annotation, variant-effect prediction | [Genome](/molecular-modeling/genome) |

## Scope Map

| Scope | Covers | Keep Separate |
| --- | --- | --- |
| Computational biology | protein, molecule, ligand, pocket, complex, structure, genome sequence | broad omics and clinical biology |
| Molecular modeling | conformer, docking, pose, force field, virtual screening, molecular property | protein sequence-only or genome-only tasks |
| Protein modeling | sequence, structure, domain, binding site, design, interaction | ligand chemistry and assay semantics |
| Structure-based modeling | pocket, protein-ligand geometry, pose, scoring, structure-aware generation | architecture definitions |
| AI method | architecture, objective, learning method, generative model, evaluation method | domain object definitions |
| Math foundation | likelihood, loss, gradient, geometry, symmetry, metric formula | workflow-specific assumptions |

## 객체에서 시작하기

Computational Biology 논문은 모델 이름보다 먼저 무엇을 모델링하는지 고정해야 합니다.

| Modeled Object | Typical Questions | Start |
| --- | --- | --- |
| Molecule / ligand | 어떤 chemical state, graph, fingerprint, conformer를 쓰는가? | [Molecules](/molecular-modeling/molecules), [Molecule](/entities/molecule), [Ligand](/entities/ligand) |
| Protein / target | sequence-only인가, structure-aware인가, 어떤 family split이 필요한가? | [Proteins](/molecular-modeling/proteins), [Protein](/entities/protein), [Target](/entities/target) |
| Pocket / complex | pocket이 known, predicted, ligand-defined, blind 중 무엇인가? | [Structure-Based Modeling](/molecular-modeling/structure-based), [Pocket](/entities/pocket), [Protein-ligand complex](/entities/protein-ligand-complex) |
| Bioactivity label | target, assay, endpoint, unit, threshold, censoring, source가 보존되는가? | [Data and Evaluation](/molecular-modeling/data-evaluation), [Target-assay-label contract](/entities/target-assay-label) |
| Genome region | broad omics가 아니라 sequence/region/variant 입력으로 다루는가? | [Genome](/molecular-modeling/genome), [Genome modeling concepts](/concepts/genome-modeling) |

## 구조 기반 문제

Structure-based modeling은 별도의 큰 덩어리로 봅니다. Docking은 그 안의 workflow입니다.

| Question | Route |
| --- | --- |
| receptor와 ligand를 어떻게 준비하는가? | [Docking workflow](/concepts/sbdd/docking-workflow), [Receptor and ligand preparation](/concepts/sbdd/receptor-ligand-preparation) |
| pose를 생성하거나 refine하는가? | [Protein-ligand docking](/molecular-modeling/structure-based/protein-ligand-docking), [Docking](/molecular-modeling/docking), [Pose generation](/concepts/sbdd/pose-generation) |
| geometry가 타당한가? | [Geometry](/molecular-modeling/geometry), [Pose quality](/concepts/sbdd/pose-quality), [PoseBusters](/papers/sbdd/posebusters) |
| score가 pose, affinity, ranking, enrichment 중 무엇을 뜻하는가? | [Scoring function](/concepts/sbdd/scoring-function), [Binding affinity](/concepts/sbdd/binding-affinity), [Virtual screening](/concepts/sbdd/virtual-screening) |
| train/test가 ligand와 protein 양쪽에서 분리되는가? | [Protein-ligand split](/concepts/sbdd/protein-ligand-split), [Leakage](/concepts/evaluation/leakage) |

## 논문을 읽을 때

| 먼저 볼 것 | 확인할 내용 | Start |
| --- | --- | --- |
| Object | molecule, protein, ligand, pocket, complex, genome region 중 무엇인가 | [Entities](/molecular-modeling/entities) |
| Representation | string, graph, fingerprint, embedding, conformer, coordinate, complex graph 중 무엇인가 | [Molecules](/molecular-modeling/molecules), [Proteins](/molecular-modeling/proteins), [Geometry](/molecular-modeling/geometry) |
| Chemical state | salt, stereo, tautomer, protonation, charge, conformer policy가 명확한가 | [Chemical state contract](/concepts/molecular-modeling/chemical-state-contract) |
| Label context | target, assay, endpoint, unit, threshold, censoring, source가 명확한가 | [Data and Evaluation](/molecular-modeling/data-evaluation), [Target-assay-label contract](/entities/target-assay-label) |
| Split unit | scaffold, protein family, complex pair, assay/source, time 중 무엇으로 나누는가 | [Data and Evaluation](/molecular-modeling/data-evaluation) |
| Evaluation claim | pose, affinity, ranking, enrichment, property, generation 중 무엇을 주장하는가 | [Docking](/molecular-modeling/docking), [Evaluation](/ai/evaluation) |

## Related

- [[ai/index|AI]]
- [[math/index|Math]]
- [[concepts/molecular-modeling/index|Molecular modeling concepts]]
- [[concepts/protein-modeling/index|Protein modeling concepts]]
- [[concepts/sbdd/index|SBDD concepts]]
- [[papers/index|Papers]]
