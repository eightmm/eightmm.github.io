---
title: Leakage
tags:
  - evaluation
  - methodology
  - data
---

# Leakage

Leakage is when information from the test set reaches the model during training, inflating reported performance that then collapses in real use. It hides in duplicate records, shared identifiers across splits, target-derived features, and preprocessing fit on the full dataset.

## Practical Checks

- Split before any fitting — scalers, encoders, and imputers learn on train only.
- Deduplicate and check for near-duplicates across train/val/test.
- Ensure grouped entities (molecule, protein, patient) never span splits.
- Audit features for any that encode the label or future information.
- If a result looks too good, suspect leakage before celebrating.

## Related

- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/index|Evaluation]]
