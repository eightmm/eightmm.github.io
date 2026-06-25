---
title: Data Preprocessing
tags:
  - machine-learning
  - data
---

# Data Preprocessing

Data preprocessing transforms raw examples into model-ready inputs. It includes cleaning, normalization, imputation, tokenization, graph construction, and split-aware fitting.

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

## Checks

- Was the split created before fitting any preprocessing step?
- Are missing values imputed using train-only statistics?
- Are molecule/protein duplicates removed before split assignment?
- Does preprocessing discard information required by the task?
- Are tokenization and graph construction deterministic and documented?

## Related

- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/architectures/graph-construction|Graph construction]]
