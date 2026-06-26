---
title: Data Preprocessing
tags:
  - machine-learning
  - data
---

# Data Preprocessing

Data preprocessing transforms raw examples into model-ready inputs. It includes cleaning, normalization, imputation, tokenization, graph construction, and split-aware fitting.

A preprocessing pipeline is a function fitted on the training set:

$$
\hat{T}
=
\operatorname{fit}(T,\mathcal{D}_{\mathrm{train}})
$$

The fitted transform is then applied unchanged:

$$
\tilde{x}
=
\hat{T}(x)
$$

This distinction matters because preprocessing can leak validation or test information just as easily as model training can.

A standard normalization fit on training data is:

$$
\mu_{\mathrm{train}}
= \frac{1}{n_{\mathrm{train}}}\sum_{i\in\mathrm{train}} x_i
$$

$$
\sigma_{\mathrm{train}}^2
= \frac{1}{n_{\mathrm{train}}}\sum_{i\in\mathrm{train}}
(x_i-\mu_{\mathrm{train}})^2
$$

$$
\tilde{x}
= \frac{x-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}+\epsilon}
$$

The same $\mu_{\mathrm{train}}$ and $\sigma_{\mathrm{train}}$ must be used for validation and test data.

## Common Operations

- Cleaning: remove invalid, corrupted, duplicated, or out-of-scope examples.
- Normalization: put numeric features on comparable scales.
- Imputation: fill missing values using train-only statistics or an explicit missingness model.
- Tokenization: map raw text, sequence, or symbolic input to discrete units.
- Featurization: compute fixed descriptors such as fingerprints, k-mers, or engineered tabular features.
- Graph construction: choose nodes, edges, features, cutoffs, and coordinate handling.
- Augmentation: create stochastic variants while preserving the label semantics.

For mean imputation:

$$
\tilde{x}_{ij}
=
\begin{cases}
x_{ij}, & x_{ij}\ \text{observed} \\
\mu_{j,\mathrm{train}}, & x_{ij}\ \text{missing}
\end{cases}
$$

The missingness indicator can be kept as an additional feature when missingness itself carries signal:

$$
m_{ij}=\mathbb{1}[x_{ij}\ \text{is missing}]
$$

## Split-Aware Rule

Every fitted preprocessing component must obey:

$$
\operatorname{fit}(T,\mathcal{D})
=
\operatorname{fit}(T,\mathcal{D}_{\mathrm{train}})
$$

not:

$$
\operatorname{fit}(T,\mathcal{D}_{\mathrm{train}}\cup
\mathcal{D}_{\mathrm{val}}\cup
\mathcal{D}_{\mathrm{test}})
$$

This applies to scalers, vocabulary builders, PCA, clustering-based features, duplicate removal, outlier thresholds, and learned featurizers.

## Reproducibility

Preprocessing is part of the experiment artifact. A result is hard to interpret if the transform version, random seed, vocabulary, descriptor settings, or graph-construction parameters are missing.

## Checks

- Was the split created before fitting any preprocessing step?
- Are missing values imputed using train-only statistics?
- Are molecule/protein duplicates removed before split assignment?
- Does preprocessing discard information required by the task?
- Are tokenization and graph construction deterministic and documented?
- Are stochastic augmentations disabled or controlled during validation and test?
- Is the same fitted transform version used for training, evaluation, and deployment?

## Related

- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/architectures/graph-construction|Graph construction]]
