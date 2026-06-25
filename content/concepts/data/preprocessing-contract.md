---
title: Preprocessing Contract
tags:
  - data
  - systems
  - reproducibility
---

# Preprocessing Contract

A preprocessing contract defines how raw records become model-ready inputs. It prevents training, evaluation, and inference from silently using different transformations.

A preprocessing pipeline is a function:

$$
z_i = P_\phi(r_i)
$$

$r_i$ is a raw record, $P_\phi$ is the preprocessing function with versioned parameters $\phi$, and $z_i$ is the model-ready representation.

## Contract Fields

- Raw input schema and required fields.
- Filtering and exclusion rules.
- Normalization, tokenization, featurization, or structure preparation.
- Missing-value policy.
- Fit-on-train-only transforms such as scalers, imputers, thresholds, or vocabularies.
- Output schema, units, shapes, masks, and valid ranges.
- Version, hash, and reproducibility record.

## Checks

- Is the same contract used for train, validation, test, and inference?
- Are learned preprocessing parameters fit only on training data?
- Does the contract preserve label semantics and units?
- Are invalid examples logged rather than silently dropped?
- Can another run reconstruct the same processed examples?
- Are filtering decisions recorded before metrics are computed?
- Does preprocessing depend only on information available at deployment time?

## Leakage Boundary

Any fitted preprocessing parameter belongs to the training split:

$$
\phi_{\mathrm{train}}
=
\operatorname{fit}(P, \mathcal{D}_{\mathrm{train}})
$$

Validation, test, and inference should only apply:

$$
z = P_{\phi_{\mathrm{train}}}(r)
$$

This applies to scalers, imputers, vocabularies, thresholds, feature selectors, molecular standardization decisions, and structure-preparation rules that are learned or chosen from data.

## Related

- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/data/data-schema|Data schema]]
- [[concepts/data/data-versioning|Data versioning]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
