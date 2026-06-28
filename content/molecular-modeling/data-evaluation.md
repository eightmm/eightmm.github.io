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

Computational biology evaluation is mostly decided before training: by example definitions, label semantics, split units, leakage controls, and benchmark design. Random splits are often too weak for molecule, protein, and complex-level claims.

$$
\hat{R}(f)
=
\frac{1}{m}\sum_{j=1}^{m}\mathcal{L}(f(x_j), y_j)
$$

This estimate is only meaningful if the test set matches the generalization claim.

## Data Contract Routes

| Question | Start | Watch |
| --- | --- | --- |
| What counts as one example? | [Example unit](/concepts/data/example-unit), [Dataset construction checklist](/concepts/data/dataset-construction-checklist) | row IDs that hide molecule, protein, assay, or complex identity |
| What does the label mean? | [Label semantics](/concepts/data/label-semantics), [Target-assay-label contract](/entities/target-assay-label), [Bioactivity label](/entities/bioactivity-label) | endpoint, unit, censoring, threshold, source mismatch |
| What preprocessing changes the object? | [Preprocessing contract](/concepts/data/preprocessing-contract), [Metadata and provenance](/concepts/data/metadata-provenance), [Chemical state contract](/concepts/molecular-modeling/chemical-state-contract) | fitting preprocessing or deduplication on full data |
| Which split supports the claim? | [Split unit](/concepts/data/split-unit), [Leakage](/concepts/evaluation/leakage), [Test-set contamination](/concepts/evaluation/test-set-contamination) | random rows overstating generalization |

## Split and Leakage Routes

| Split Axis | Start | Use For |
| --- | --- | --- |
| Molecule series | [Scaffold split](/concepts/evaluation/scaffold-split) | new-chemistry claims and analog leakage checks |
| Protein family | [Protein family split](/concepts/evaluation/protein-family-split) | new-target or sequence generalization claims |
| Protein-ligand pair | [Protein-ligand split](/concepts/sbdd/protein-ligand-split) | interaction and docking generalization |
| Structure template | [Template leakage](/concepts/sbdd/template-leakage) | structure prediction, docking, pocket tasks |
| Assay/source | [Assay harmonization](/concepts/evaluation/assay-harmonization) | multi-source activity labels |

## Claim to Split Map

| Claim | Stronger Split | Weak Split Risk |
| --- | --- | --- |
| New molecule within known targets | scaffold or chemical-series split | random split can memorize close analogs |
| New protein target | protein-family or sequence-identity split | random target rows can share homologs |
| New protein-ligand pair | pair split plus scaffold and protein controls | pair split alone may preserve both ligand and protein neighborhoods |
| New assay/source | source or temporal split | assay-specific artifacts can leak across rows |
| New structure template | template-aware structure split | homologous structures can make pose tasks too easy |

## Evaluation Routes

| Need | Start | Check |
| --- | --- | --- |
| Protocol boundary | [Evaluation](/concepts/evaluation), [Evaluation protocol](/concepts/evaluation/evaluation-protocol), [Evaluation set design](/concepts/evaluation/evaluation-set-design) | train/validation/test roles and final selection rule |
| Claim support | [Claim-evidence boundary](/concepts/evaluation/claim-evidence-boundary), [Benchmark claim contract](/concepts/evaluation/benchmark-claim-contract), [Benchmark intake](/concepts/data/benchmark-intake) | what the reported number can actually prove |
| Metric choice | [Metric selection](/concepts/evaluation/metric-selection), [Objective-metric alignment](/concepts/machine-learning/objective-metric-alignment) | optimized loss versus reported decision metric |
| Reliability | [Calibration](/concepts/evaluation/calibration), [Uncertainty estimation](/concepts/evaluation/uncertainty-estimation), [Applicability domain](/concepts/evaluation/applicability-domain) | confidence under distribution shift |
| Benchmark traps | [Negative set](/concepts/evaluation/negative-set), [Activity cliff](/concepts/evaluation/activity-cliff), [Boltzmann ceiling analysis](/concepts/evaluation/boltzmann-ceiling) | false negatives, cliffs, indistinguishable labels |
| Repeated paper patterns | [Computational Biology paper claim patterns](/molecular-modeling/paper-claim-patterns) | property, activity, docking, generation, protein design, genome task |

## Computational Biology Evidence Package

For computational biology papers, a model score is not enough. Record the modeled object and the measurement context before the metric.

| Field | Required Detail |
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

For structure-based benchmarks, also state whether the known ligand pose, close analogs, templates, or pocket definition are available at inference time.

## Baseline Ladder

복잡한 모델은 항상 적절한 단순 baseline과 비교해야 합니다.

| Task | Useful Baseline |
| --- | --- |
| molecule property | fingerprint + linear/tree model |
| target-conditioned activity | ligand-only baseline, target-only baseline, nearest-neighbor baseline |
| protein sequence task | homology, family, or sequence-similarity baseline |
| docking / pose | classical docking, minimized pose, known conformer baseline |
| virtual screening | simple similarity, docking score, random/ranked decoy control |
| generation | validity/diversity filter plus nearest-neighbor analysis |

## Benchmark Trap Map

| Trap | Ask First | Start |
| --- | --- | --- |
| assumed negative | is `inactive` measured, censored, sampled, or missing? | [Negative set](/concepts/evaluation/negative-set) |
| analog cliff | do similar molecules have assay-compatible but very different labels? | [Activity cliff](/concepts/evaluation/activity-cliff) |
| out-of-domain example | is the test object far from training support on the relevant axis? | [Applicability domain](/concepts/evaluation/applicability-domain) |
| mixed assays | are endpoints, units, constructs, sources, and censoring compatible? | [Assay harmonization](/concepts/evaluation/assay-harmonization) |
| physical or assay ceiling | is the effect size below distinguishability or label noise? | [Boltzmann ceiling analysis](/concepts/evaluation/boltzmann-ceiling) |

## Checks

- What is one example: molecule, protein, assay record, protein-ligand complex, or generated pose?
- What is the label direction, unit, censoring rule, and assay context?
- Which near-duplicates are separated before split construction?
- Does the metric reflect the intended decision, not only benchmark convenience?

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/entities|Entities]]
- [[molecular-modeling/interactions|Interaction modeling]]
- [[molecular-modeling/paper-claim-patterns|Computational Biology paper claim patterns]]
- [[ai/evaluation|Evaluation]]
