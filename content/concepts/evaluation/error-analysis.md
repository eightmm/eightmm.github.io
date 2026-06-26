---
title: Error Analysis
tags:
  - evaluation
  - methodology
  - diagnostics
---

# Error Analysis

Error analysis studies where and why a model fails. It turns a single aggregate metric into actionable failure categories using [[concepts/evaluation/failure-mode-taxonomy|Failure mode taxonomy]].

For a dataset partitioned into groups $G_1,\ldots,G_K$, group risk is:

$$
\hat{R}_k(f)
=
\frac{1}{|G_k|}
\sum_{(x_i,y_i)\in G_k}
\mathcal{L}(f(x_i),y_i)
$$

Aggregate performance can hide large group-specific failures.

## Error Decomposition

For a metric or loss $m_i$, slice-level performance is:

$$
\hat{M}_k
=
\frac{1}{|G_k|}
\sum_{i\in G_k} m_i
$$

The worst-slice gap is:

$$
\Delta_{\mathrm{slice}}
=
\max_k \hat{R}_k - \hat{R}_{\mathrm{all}}
$$

For classification, separate false positives and false negatives:

$$
\mathrm{FPR}_k
=
\frac{\mathrm{FP}_k}{\mathrm{FP}_k+\mathrm{TN}_k},
\qquad
\mathrm{FNR}_k
=
\frac{\mathrm{FN}_k}{\mathrm{FN}_k+\mathrm{TP}_k}
$$

For ranking or screening, inspect top-ranked false positives and missed actives, not only aggregate enrichment.

## Common Slices

- Class, label range, or target type.
- Data source, time, protocol, or collection batch.
- Modality, sequence length, resolution, graph size, or structure quality.
- Molecular scaffold, protein family, pocket type, assay type, or activity cliff.
- Confidence level, uncertainty bucket, or calibration bin.
- Accepted vs abstained examples in [[concepts/evaluation/selective-prediction|selective prediction]] workflows.

## Failure Attribution

| Failure Source | Diagnostic Question | Follow-Up |
| --- | --- | --- |
| Data | Is the label wrong, noisy, censored, or inconsistent? | audit label semantics and provenance |
| Split | Is the test example near a train example or outside support? | check leakage and applicability domain |
| Representation | Did preprocessing remove needed information? | inspect features, tokenization, conformers, structures |
| Objective | Does the training loss optimize the wrong behavior? | compare objective and metric |
| Architecture | Is the inductive bias mismatched to the object? | inspect modality and symmetry requirements |
| Metric | Does the metric hide the user-facing failure? | add slice, calibration, or ranking diagnostics |
| System | Did batching, precision, decoding, or timeout change behavior? | reproduce with controlled inference settings |

## Computational Biology Error Slices

| Task | Useful Slices |
| --- | --- |
| Molecular property | scaffold, molecular weight, charge, stereochemistry, assay source |
| Activity prediction | target family, endpoint, unit, censoring, active threshold |
| Docking | receptor state, ligand flexibility, pose RMSD bucket, failed docking |
| Protein modeling | sequence identity, domain family, length, structure resolution, template availability |
| Virtual screening | active prevalence, decoy source, library subset, top-K budget |

## Error Analysis Output

An error analysis section should usually include:

| Item | Description |
| --- | --- |
| Top failure slices | the largest or most important groups where performance degrades |
| Representative examples | a few examples with input, prediction, label, and context |
| Competing explanations | data issue, model issue, objective mismatch, metric mismatch |
| Next experiment | the smallest check that would distinguish explanations |

## Checks

- Which examples dominate the error?
- Are errors concentrated in a group the benchmark underweights?
- Are false positives and false negatives caused by different mechanisms?
- Does the error reflect data quality, model architecture, objective, or metric mismatch?
- Can the failure category guide the next experiment?
- Does the slice have enough examples for the claim?
- Are examples selected before looking at the desired conclusion?

## Related

- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/failure-mode-taxonomy|Failure mode taxonomy]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/selective-prediction|Selective prediction]]
- [[concepts/evaluation/robustness|Robustness]]
- [[concepts/machine-learning/objective-metric-alignment|Objective-metric alignment]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
