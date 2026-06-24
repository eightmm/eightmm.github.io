---
title: Scaffold Split
tags:
  - evaluation
  - methodology
  - cheminformatics
---

# Scaffold Split

A scaffold split groups molecules by their core structure (e.g. Bemis–Murcko scaffold) and keeps each scaffold entirely in one split. It estimates how a model generalizes to chemotypes it has not seen, unlike a random split that scatters close analogs across train and test.

## Practical Checks

- Compute scaffolds consistently and assign whole scaffold groups to a single split.
- Expect lower, more honest scores than random splits — that is the point.
- Watch for tiny test scaffolds that make metrics noisy.
- Combine with duplicate/near-duplicate removal to avoid hidden leakage.
- Report the split method explicitly; "test accuracy" is meaningless without it.

## Related

- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/index|Evaluation]]
