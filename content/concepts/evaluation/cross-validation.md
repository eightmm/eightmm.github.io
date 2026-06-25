---
title: Cross-Validation
tags:
  - evaluation
  - validation
  - methodology
---

# Cross-Validation

Cross-validation estimates performance by repeatedly splitting data into train and validation folds. It is useful when data is limited, but it does not replace a final held-out test set for a publication-quality claim.

In $K$-fold cross-validation, the data is partitioned into folds $\mathcal{D}_1,\ldots,\mathcal{D}_K$. The model is trained on all but one fold and evaluated on the held-out fold:

$$
\hat{M}_{\mathrm{CV}}
=
\frac{1}{K}
\sum_{k=1}^{K}
M(f_{-k},\mathcal{D}_k)
$$

where $f_{-k}$ is trained without fold $k$.

## Key Ideas

- Cross-validation reduces dependence on one arbitrary validation split.
- It can still leak if near-duplicates, scaffolds, protein families, time, or assay batches cross folds.
- Hyperparameter tuning inside cross-validation requires nesting or a separate final test set.
- For grouped data, folds should be grouped by the entity that defines generalization.

## Practical Checks

- Are folds random, stratified, scaffold-based, family-based, temporal, or grouped?
- Is preprocessing fitted only on the training portion of each fold?
- Are hyperparameters selected without peeking at test data?
- Is a final untouched test set reserved for the main claim?

## Related

- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/metric|Metric]]
