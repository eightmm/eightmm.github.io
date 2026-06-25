---
title: Tree-Based Model
tags:
  - machine-learning
---

# Tree-Based Model

A tree-based model splits the feature space into regions and assigns predictions inside each region. Decision trees, random forests, and gradient-boosted trees belong to this family.

## Strengths

- Handles nonlinear feature interactions.
- Works well on tabular data.
- Requires less feature scaling than many linear or kernel methods.

## Limits

- Single trees can overfit.
- Extrapolation outside the observed feature range is weak.
- Large ensembles can be harder to interpret than a single tree.

## Related

- [[concepts/machine-learning/ensemble-method|Ensemble method]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/evaluation/index|Evaluation]]
