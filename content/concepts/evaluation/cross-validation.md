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

## Fold Unit

The fold assignment function should be written explicitly:

$$
g(u_i) \in \{1,\ldots,K\}
$$

where $u_i$ is the example unit and $g$ maps examples to folds. If the claim is scaffold generalization, $g$ should depend on scaffold. If the claim is protein-family generalization, $g$ should depend on family or sequence identity group.

| Claim | Fold Unit |
| --- | --- |
| IID interpolation | random or stratified example |
| chemical generalization | scaffold, series, time, or molecule identity group |
| protein generalization | family, target, sequence identity cluster |
| complex generalization | paired ligand and protein grouping |
| assay/source robustness | assay, source, lab, batch, endpoint group |
| temporal deployment | time-based fold or final time holdout |

## Nested Selection

If hyperparameters are tuned inside cross-validation, use nested CV:

$$
\hat{M}_{\mathrm{nested}}
=
\frac{1}{K}
\sum_{k=1}^{K}
M\left(f_{\lambda_k^\*,-k}, \mathcal{D}_k\right)
$$

where $\lambda_k^\*$ is selected using only the training portion of outer fold $k$. Reusing the outer validation fold for both selection and final reporting overestimates performance.

## Practical Checks

- Are folds random, stratified, scaffold-based, family-based, temporal, or grouped?
- Is preprocessing fitted only on the training portion of each fold?
- Are hyperparameters selected without peeking at test data?
- Is a final untouched test set reserved for the main claim?
- Are near-duplicates, scaffolds, homologs, assays, or time groups kept within the same fold?
- Is the reported uncertainty across folds, seeds, or examples?

## Related

- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/machine-learning/model-selection|Model selection]]
