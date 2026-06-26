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
- Activity cliffs define local extrapolation, not just global regression error.

## Cliff Definition Choices

| Choice | Typical Option | Risk |
| --- | --- | --- |
| molecular similarity | ECFP Tanimoto, MCS similarity, scaffold-neighborhood rule | different fingerprints produce different cliff sets |
| activity scale | pIC50, pKi, pKd, Delta G, binary threshold | mixed endpoints can create artificial cliffs |
| activity gap | fixed threshold, assay-noise-aware threshold, replicate-aware threshold | threshold below measurement noise overstates cliffs |
| target context | same target, same assay, same protein construct | cross-assay comparisons can invent cliffs |
| split policy | keep analog series grouped or explicitly evaluate cliff transfer | random split can leak cliff neighborhoods |

## Evaluation Pattern

| Question | Required Evidence |
| --- | --- |
| Does the model handle local SAR? | metrics stratified on cliff vs non-cliff examples |
| Is the cliff real? | same target, same assay context, compatible units, replicate handling |
| Is the split honest? | near-neighbor or scaffold controls before train/test assignment |
| Is the error meaningful? | compare activity gap to assay noise or confidence interval |
| Does improvement matter? | paired comparison on the same cliff subset |

## Checks

- Are cliff pairs detected before reporting final metrics?
- Are metrics stratified on cliff and non-cliff subsets?
- Does the split policy keep near-duplicate analog series honest?
- Are assay differences causing apparent cliffs?
- Is the similarity threshold stated rather than implied?
- Is the activity threshold larger than expected measurement noise?
- Are cliff examples overrepresented in validation selection?

## Related

- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/molecular-modeling/molecular-similarity|Molecular similarity]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[entities/dataset|Dataset]]
