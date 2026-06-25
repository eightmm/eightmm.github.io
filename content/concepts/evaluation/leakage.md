---
title: Leakage
tags:
  - evaluation
  - methodology
  - data
---

# Leakage

Leakage is when information from the test set reaches the model during training, inflating reported performance that then collapses in real use. It hides in duplicate records, shared identifiers across splits, target-derived features, and preprocessing fit on the full dataset.

The split should satisfy an independence assumption:

$$
\mathcal{D}_{\mathrm{train}}
\cap_{\mathrm{information}}
\mathcal{D}_{\mathrm{test}}
= \varnothing
$$

The subscript means exact row overlap is not enough; near-duplicates, shared groups, and target-derived features can also transmit information.

## Practical Checks

- Split before any fitting — scalers, encoders, and imputers learn on train only.
- Deduplicate and check for near-duplicates across train/val/test.
- Ensure grouped entities (molecule, protein, patient) never span splits.
- Audit features for any that encode the label or future information.
- If a result looks too good, suspect leakage before celebrating.

## Common Leakage Channels

- Duplicate or near-duplicate examples across splits.
- Shared scaffold, protein family, subject, source, or time period across splits.
- Preprocessing fit on the full dataset before splitting.
- Feature selection, imputation, normalization, or vocabulary building that sees test data.
- Target-derived features, post-outcome metadata, or future information.
- Test-set iteration during model selection.

## Leakage Test

For a split function $s$ and information unit $u$, a basic grouped-leakage condition is:

$$
s(x_i)\ne s(x_j)
\Rightarrow
u(x_i)\ne u(x_j)
$$

This condition must be checked for every unit that could transmit target-relevant information.

## Related

- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/data/data-schema|Data schema]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/index|Evaluation]]
