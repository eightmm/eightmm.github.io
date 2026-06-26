---
title: Data and Evaluation
aliases:
  - bio-ai/data-evaluation
tags:
  - bio
  - evaluation
---


# Data and Evaluation

Molecular modeling evaluation is mostly decided before training: by example definitions, label semantics, split units, leakage controls, and benchmark design. Random splits are often too weak for molecule, protein, and complex-level claims.

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
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]

## Molecular Modeling Evidence Package

For molecular modeling papers, a model score is not enough. Record the modeled object and the measurement context before the metric.

| Field | Required Detail |
| --- | --- |
| Example unit | molecule, protein, assay record, complex, pose, generated sample, or genomic region |
| Label semantics | endpoint, direction, unit, threshold, censoring, replicate aggregation, and source |
| Preprocessing | molecule standardization, protein/structure cleaning, sequence filtering, coordinate source |
| Split unit | scaffold, protein family, complex pair, assay/source, time, or template-aware split |
| Baseline | fingerprint/tree model, sequence similarity, docking baseline, or task-specific simple model |
| Metric | primary decision metric plus diagnostics for calibration, uncertainty, or failure modes |
| Leakage check | duplicate, scaffold, homolog, template, assay/source, or preprocessing leakage |

For structure-based benchmarks, also state whether the known ligand pose, close analogs, templates, or pocket definition are available at inference time.

## Checks

- What is one example: molecule, protein, assay record, protein-ligand complex, or generated pose?
- What is the label direction, unit, censoring rule, and assay context?
- Which near-duplicates are separated before split construction?
- Does the metric reflect the intended decision, not only benchmark convenience?

## Related

- [[bio/index|Molecular Modeling]]
- [[bio/entities|Entities]]
- [[ai/evaluation|Evaluation]]
