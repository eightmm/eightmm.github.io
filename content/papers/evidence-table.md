---
title: Evidence Table
tags:
  - papers
  - methodology
  - evaluation
---

# Evidence Table

An evidence table maps each paper claim to the experiment, metric, baseline, and limitation that support it. It prevents a paper note from becoming a list of ungrounded impressions.

The core relation is:

$$
\text{claim}
\rightarrow
\text{evidence}
\rightarrow
\text{scope}
\rightarrow
\text{limit}
$$

## Suggested Columns

| Column | Meaning |
| --- | --- |
| Claim | The narrow statement being evaluated. |
| Evidence | Figure, table, proof, benchmark, or experiment supporting it. |
| Task | What task the evidence actually tests. |
| Data | Dataset, benchmark, split, and preprocessing context. |
| Metric | Number used to support the claim. |
| Baseline | Comparison that makes the metric meaningful. |
| Uncertainty | Confidence interval, seed variance, or missing uncertainty. |
| Limitation | Why the claim may not generalize. |
| Wiki links | Reusable concepts updated by the claim. |

## Evidence Strength

A rough evidence strength score can be written as:

$$
S
=
\mathbf{1}_{\mathrm{metric}}
+ \mathbf{1}_{\mathrm{baseline}}
+ \mathbf{1}_{\mathrm{split}}
+ \mathbf{1}_{\mathrm{uncertainty}}
+ \mathbf{1}_{\mathrm{ablation}}
$$

This is not a scientific score; it is a checklist for whether the paper gives enough information to interpret a claim.

## Checks

- Is every important claim tied to a concrete result?
- Does the metric match the task?
- Is the baseline strong enough?
- Does the split support the stated generalization claim?
- Are uncertainty and failure cases reported?
- Is the claim phrased no broader than the evidence?

## Related

- [[papers/claim-extraction|Claim extraction]]
- [[papers/paper-note-format|Paper note format]]
- [[papers/paper-review-workflow|Paper review workflow]]
- [[papers/limitation-taxonomy|Limitation taxonomy]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/baseline|Baseline]]
