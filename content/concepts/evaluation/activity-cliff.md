---
title: Activity Cliff
tags:
  - evaluation
  - molecular-modeling
  - dataset
---

# Activity Cliff

An activity cliff is a pair of similar molecules with a large difference in measured activity. Activity cliffs expose where smooth similarity assumptions fail.

A simple cliff condition is:

$$
\operatorname{sim}(M_i,M_j) \ge \tau_s
\quad\text{and}\quad
|a_i-a_j| \ge \tau_a
$$

where $\operatorname{sim}$ is molecular similarity, $a_i$ and $a_j$ are activity values, and $\tau_s,\tau_a$ are thresholds.

## Why It Matters

- Similarity-based models often struggle on cliffs.
- Random splits can scatter cliff pairs across train and test.
- Aggregate metrics can hide failure on hard local structure-activity regions.

## Checks

- Are cliff pairs detected before reporting final metrics?
- Are metrics stratified on cliff and non-cliff subsets?
- Does the split policy keep near-duplicate analog series honest?
- Are assay differences causing apparent cliffs?

## Related

- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[entities/dataset|Dataset]]
