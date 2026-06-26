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

## Schema Boundary

Tabular modeling depends on a schema:

$$
\mathcal{S}
=
\{(c_j,\ \text{type}_j,\ \text{unit}_j,\ \text{missing policy}_j)\}_{j=1}^{d}
$$

where each column has a type, unit, allowed values, and missingness interpretation. A column named the same way across datasets may not mean the same thing.

## Preprocessing Contract

The preprocessing map should be fit on training data only:

$$
\phi_{\mathrm{train}}
=
\operatorname{fit}(\mathcal{D}_{\mathrm{train}})
$$

$$
X_{\mathrm{val/test}}
=
\phi_{\mathrm{train}}(\mathcal{D}_{\mathrm{val/test}})
$$

This applies to imputation, scaling, encoding, feature selection, target transforms, and rare-category handling.

## Leakage Risks

- IDs, timestamps, or metadata encode the answer.
- Aggregates are computed using validation/test rows.
- Multiple rows from the same entity appear across splits.
- Missingness itself reveals collection process or outcome.
- Post-outcome measurements are used as input features.

## Practical Checks

- What is one row and what is one label?
- Are categorical variables encoded consistently across train and test?
- Are missing values meaningful or merely absent?
- Do any columns leak target, time, batch, user, or split information?
- Is the split grouped by entity when rows are not independent?
- Are all preprocessing statistics fit only on training rows?
- Are units, thresholds, and target transforms recorded?
- Is a simple baseline strong enough to challenge more complex models?

## Related

- [[concepts/data/data-schema|Data schema]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/data/missing-data|Missing data]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/machine-learning/tree-based-model|Tree-based model]]
- [[concepts/data/data-curation|Data curation]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/evaluation/leakage|Leakage]]
