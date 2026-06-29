---
title: Data and Evaluation
aliases:
  - computational-biology/data-evaluation
  - bio/data-evaluation
tags:
  - computational-biology
  - evaluation
---


# Data and Evaluation

Computational biology evaluation은 대부분 training 전에 결정됩니다. Example definition, label semantics, split unit, leakage control, benchmark design이 성능 claim의 범위를 정합니다. Random split은 molecule, protein, complex-level claim에는 약한 경우가 많습니다.

$$
\hat{R}(f)
=
\frac{1}{m}\sum_{j=1}^{m}\mathcal{L}(f(x_j), y_j)
$$

이 estimate는 test set이 generalization claim과 맞을 때만 의미가 있습니다.

## Data Contract Routes

| Question | Start | Watch |
| --- | --- | --- |
| 무엇이 one example인가? | [Example unit](/concepts/data/example-unit), [Dataset construction checklist](/concepts/data/dataset-construction-checklist) | molecule, protein, assay, complex identity를 숨기는 row ID |
| label은 무엇을 뜻하는가? | [Label semantics](/concepts/data/label-semantics), [Target-assay-label contract](/entities/target-assay-label), [Bioactivity label](/entities/bioactivity-label) | endpoint, unit, censoring, threshold, source mismatch |
| preprocessing이 object를 어떻게 바꾸는가? | [Preprocessing contract](/concepts/data/preprocessing-contract), [Metadata and provenance](/concepts/data/metadata-provenance), [Chemical state contract](/concepts/molecular-modeling/chemical-state-contract) | full data에서 preprocessing 또는 deduplication을 fit하는 경우 |
| 어떤 split이 claim을 지지하는가? | [Split unit](/concepts/data/split-unit), [Leakage](/concepts/evaluation/leakage), [Test-set contamination](/concepts/evaluation/test-set-contamination) | random row split이 generalization을 과장하는 경우 |

## Split and Leakage Routes

| Split axis | Start | Use for |
| --- | --- | --- |
| Molecule series | [Scaffold split](/concepts/evaluation/scaffold-split) | new-chemistry claim과 analog leakage check |
| Protein family | [Protein family split](/concepts/evaluation/protein-family-split) | new-target 또는 sequence generalization claim |
| Protein-ligand pair | [Protein-ligand split](/concepts/sbdd/protein-ligand-split) | interaction과 docking generalization |
| Structure template | [Template leakage](/concepts/sbdd/template-leakage) | structure prediction, docking, pocket task |
| Assay/source | [Assay harmonization](/concepts/evaluation/assay-harmonization) | multi-source activity label |

## Claim to Split Map

| Claim | 더 강한 split | 약한 split의 risk |
| --- | --- | --- |
| Known target 안의 new molecule | scaffold 또는 chemical-series split | random split이 close analog를 memorize할 수 있음 |
| New protein target | protein-family 또는 sequence-identity split | random target row가 homolog를 공유할 수 있음 |
| New protein-ligand pair | pair split + scaffold/protein control | pair split만으로는 ligand와 protein neighborhood가 남을 수 있음 |
| New assay/source | source 또는 temporal split | assay-specific artifact가 row 사이로 leak될 수 있음 |
| New structure template | template-aware structure split | homologous structure가 pose task를 너무 쉽게 만들 수 있음 |

## Evaluation Routes

| Need | Start | Check |
| --- | --- | --- |
| Protocol boundary | [Evaluation](/concepts/evaluation), [Evaluation protocol](/concepts/evaluation/evaluation-protocol), [Evaluation set design](/concepts/evaluation/evaluation-set-design) | train/validation/test role과 final selection rule |
| Claim support | [Claim-evidence boundary](/concepts/evaluation/claim-evidence-boundary), [Benchmark claim contract](/concepts/evaluation/benchmark-claim-contract), [Benchmark intake](/concepts/data/benchmark-intake) | reported number가 실제로 증명할 수 있는 것 |
| Metric choice | [Metric selection](/concepts/evaluation/metric-selection), [Objective-metric alignment](/concepts/machine-learning/objective-metric-alignment) | optimized loss와 reported decision metric의 차이 |
| Reliability | [Calibration](/concepts/evaluation/calibration), [Uncertainty estimation](/concepts/evaluation/uncertainty-estimation), [Applicability domain](/concepts/evaluation/applicability-domain) | distribution shift 아래 confidence |
| Benchmark traps | [Negative set](/concepts/evaluation/negative-set), [Activity cliff](/concepts/evaluation/activity-cliff), [Boltzmann ceiling analysis](/concepts/evaluation/boltzmann-ceiling) | false negative, cliff, indistinguishable label |
| Repeated paper patterns | [Computational Biology paper claim patterns](/molecular-modeling/paper-claim-patterns) | property, activity, docking, generation, protein design, genome task |

## Computational Biology Evidence Package

Computational biology paper에서는 model score만으로 부족합니다. Metric보다 먼저 modeled object와 measurement context를 기록해야 합니다.

| Field | Required detail |
| --- | --- |
| Example unit | molecule, protein, assay record, complex, pose, generated sample, or genomic region |
| Label semantics | endpoint, direction, unit, threshold, censoring, replicate aggregation, and source |
| Preprocessing | molecule standardization, protein/structure cleaning, sequence filtering, coordinate source |
| Chemical state | salt, stereo, tautomer, protonation, charge, conformer, and cache policy |
| Split unit | scaffold, protein family, complex pair, assay/source, time, or template-aware split |
| Baseline | fingerprint/tree model, sequence similarity, docking baseline, or task-specific simple model |
| Metric | primary decision metric plus diagnostics for calibration, uncertainty, or failure modes |
| Benchmark claim | what the reported score can support after split, metric, baseline, and uncertainty are checked |
| Objective-metric alignment | whether the optimized loss supports the reported metric and claimed utility |
| Leakage check | duplicate, scaffold, homolog, template, assay/source, or preprocessing leakage |
| Benchmark traps | negative construction, activity cliffs, applicability domain, assay harmonization, and measurement ceiling |

Structure-based benchmark에서는 known ligand pose, close analog, template, pocket definition이 inference time에 사용 가능한지도 적어야 합니다.

## Baseline Ladder

복잡한 모델은 항상 적절한 단순 baseline과 비교해야 합니다.

| Task | Useful baseline |
| --- | --- |
| molecule property | fingerprint + linear/tree model |
| target-conditioned activity | ligand-only baseline, target-only baseline, nearest-neighbor baseline |
| protein sequence task | homology, family, or sequence-similarity baseline |
| docking / pose | classical docking, minimized pose, known conformer baseline |
| virtual screening | simple similarity, docking score, random/ranked decoy control |
| generation | validity/diversity filter plus nearest-neighbor analysis |

## Benchmark Trap Map

| Trap | 먼저 물을 것 | Start |
| --- | --- | --- |
| assumed negative | `inactive`가 measured, censored, sampled, missing 중 무엇인가? | [Negative set](/concepts/evaluation/negative-set) |
| analog cliff | similar molecule들이 assay-compatible하지만 매우 다른 label을 갖는가? | [Activity cliff](/concepts/evaluation/activity-cliff) |
| out-of-domain example | test object가 relevant axis에서 training support와 먼가? | [Applicability domain](/concepts/evaluation/applicability-domain) |
| mixed assays | endpoint, unit, construct, source, censoring이 compatible한가? | [Assay harmonization](/concepts/evaluation/assay-harmonization) |
| physical or assay ceiling | effect size가 distinguishability 또는 label noise보다 작은가? | [Boltzmann ceiling analysis](/concepts/evaluation/boltzmann-ceiling) |

## Checks

- one example은 molecule, protein, assay record, protein-ligand complex, generated pose 중 무엇인가?
- label direction, unit, censoring rule, assay context는 무엇인가?
- split construction 전에 어떤 near-duplicate를 분리했는가?
- metric이 benchmark convenience만이 아니라 intended decision을 반영하는가?

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/entities|Entities]]
- [[molecular-modeling/interactions|Interaction modeling]]
- [[molecular-modeling/paper-claim-patterns|Computational Biology paper claim patterns]]
- [[ai/evaluation|Evaluation]]
