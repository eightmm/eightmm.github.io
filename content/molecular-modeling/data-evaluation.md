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

## Data Contracts

- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/molecular-modeling/chemical-state-contract|Chemical state contract]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[entities/bioactivity-label|Bioactivity label]]

## Splits and Leakage

- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/test-set-contamination|Test-set contamination]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[concepts/sbdd/template-leakage|Template leakage]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]

## Claim to Split Map

| Claim | Stronger Split | Weak Split Risk |
| --- | --- | --- |
| New molecule within known targets | scaffold or chemical-series split | random split can memorize close analogs |
| New protein target | protein-family or sequence-identity split | random target rows can share homologs |
| New protein-ligand pair | pair split plus scaffold and protein controls | pair split alone may preserve both ligand and protein neighborhoods |
| New assay/source | source or temporal split | assay-specific artifacts can leak across rows |
| New structure template | template-aware structure split | homologous structures can make pose tasks too easy |

## Evaluation

- [[concepts/evaluation/index|Evaluation]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/evaluation-set-design|Evaluation set design]]
- [[concepts/evaluation/claim-evidence-boundary|Claim-evidence boundary]]
- [[concepts/data/benchmark-intake|Benchmark intake]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[concepts/machine-learning/objective-metric-alignment|Objective-metric alignment]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/boltzmann-ceiling|Boltzmann ceiling analysis]]
- [[molecular-modeling/paper-claim-patterns|Computational Biology paper claim patterns]]

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
- [[molecular-modeling/paper-claim-patterns|Computational Biology paper claim patterns]]
- [[ai/evaluation|Evaluation]]
