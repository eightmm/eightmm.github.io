---
title: Tabular
tags:
  - modalities
  - tabular
  - data
---

# Tabular

Tabular data represents examples as rows and fields as columns. It appears in assay tables, benchmark metadata, experiment logs, hardware measurements, user studies, and operational dashboards.

A tabular dataset can be written as:

$$
X\in\mathbb{R}^{n\times d},
\qquad
y\in\mathcal{Y}^{n}
$$

where each row is an example and each column is a feature after preprocessing.

Feature preprocessing maps raw columns into model-ready inputs:

$$
\phi(x)
=
[\phi_1(x_1),\ldots,\phi_d(x_d)]
$$

## Key Ideas

- Tabular data mixes numeric, categorical, ordinal, missing, and free-text fields.
- Feature leakage is common when columns encode future information, identifiers, or post-outcome measurements.
- Strong baselines often include linear models, tree-based models, and simple feature engineering.
- Metadata quality can matter more than architecture choice.

## Practical Checks

- What is one row and what is one label?
- Are categorical variables encoded consistently across train and test?
- Are missing values meaningful or merely absent?
- Do any columns leak target, time, batch, user, or split information?
- Is the split grouped by entity when rows are not independent?

## Related

- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/machine-learning/tree-based-model|Tree-based model]]
- [[concepts/data/data-curation|Data curation]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/evaluation/leakage|Leakage]]
