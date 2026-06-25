---
title: Missing Data
tags:
  - data
  - labels
  - evaluation
---

# Missing Data

Missing data means an input feature, label, metadata field, modality, assay result, or measurement is unavailable. It is not automatically zero, negative, inactive, irrelevant, or safe to ignore.

For a variable $X$ and missingness indicator $M$:

$$
M =
\begin{cases}
1 & \text{if } X \text{ is missing} \\
0 & \text{if } X \text{ is observed}
\end{cases}
$$

The missingness process matters because:

$$
P(M=1\mid X,Y)
$$

can depend on the value itself, the label, source, time, assay, cost, or collection policy.

## Missingness Types

- MCAR: missing completely at random; missingness is independent of observed and unobserved variables.
- MAR: missing at random conditional on observed variables.
- MNAR: missingness depends on unobserved variables or the missing value itself.

In ML datasets, MNAR is common. For example, a measurement may be absent because it was expensive, low priority, hard to run, or not expected to be interesting.

## Missing Label Is Not Negative Label

The most common failure is:

$$
y_i \text{ missing}
\not\Rightarrow
y_i = 0
$$

This matters in retrieval, recommendation, virtual screening, bioactivity prediction, preference data, and weak supervision. Treating unobserved positives as negatives can create false negative labels and distort evaluation.

## Imputation

Imputation fills missing values using a rule or model:

$$
\tilde{x}_{ij}
=
\operatorname{impute}(x_{ij}, \mathcal{D}_{\mathrm{train}})
$$

The imputer must be fit on training data only. Fitting imputation on the full dataset can leak validation or test information.

## Checks

- What is missing: input, label, metadata, modality, assay, or target context?
- Is missingness MCAR, MAR, or likely MNAR?
- Are missing labels treated as unknown rather than negative?
- Is imputation fit only on the training split?
- Are missingness indicators useful features, leakage channels, or both?
- Does evaluation report performance on missingness-defined slices?

## Related

- [[concepts/modalities/missing-modality|Missing modality]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/data/censored-label|Censored label]]
- [[concepts/data/weak-label|Weak label]]
- [[concepts/data/sampling-bias|Sampling bias]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/evaluation/leakage|Leakage]]
- [[entities/dataset|Dataset]]
