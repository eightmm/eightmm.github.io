---
title: Error Analysis
tags:
  - evaluation
  - methodology
  - diagnostics
---

# Error Analysis

Error analysis studies where and why a model fails. It turns a single aggregate metric into actionable failure categories.

For a dataset partitioned into groups $G_1,\ldots,G_K$, group risk is:

$$
\hat{R}_k(f)
=
\frac{1}{|G_k|}
\sum_{(x_i,y_i)\in G_k}
\mathcal{L}(f(x_i),y_i)
$$

Aggregate performance can hide large group-specific failures.

## Common Slices

- Class, label range, or target type.
- Data source, time, protocol, or collection batch.
- Modality, sequence length, resolution, graph size, or structure quality.
- Molecular scaffold, protein family, pocket type, assay type, or activity cliff.
- Confidence level, uncertainty bucket, or calibration bin.

## Checks

- Which examples dominate the error?
- Are errors concentrated in a group the benchmark underweights?
- Are false positives and false negatives caused by different mechanisms?
- Does the error reflect data quality, model architecture, objective, or metric mismatch?
- Can the failure category guide the next experiment?

## Related

- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/robustness|Robustness]]
- [[papers/paper-review-workflow|Paper review workflow]]
