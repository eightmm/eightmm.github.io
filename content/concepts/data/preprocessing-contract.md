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

## By Modality

| Input Type | Contract Must State | Common Failure |
| --- | --- | --- |
| text | normalization, tokenizer, truncation, special tokens, document boundaries | train and inference use different tokenization or chunking |
| image/video | resize, crop, frame sampling, color normalization, augmentation policy | evaluation uses easier crops than deployment |
| molecule | salt handling, tautomer, protonation, stereo, charge, canonicalization, conformer policy | duplicate chemical states cross the split |
| protein sequence | alphabet, unknown residues, domain extraction, MSA policy, sequence clustering | homologs or MSA-derived information leak across split |
| 3D structure | coordinate source, unit, frame, missing atoms, alternate locations, pocket definition | known ligand pose or template context leaks into preprocessing |
| genome sequence | reference build, coordinate convention, strand, windowing, variant normalization | overlapping windows or locus context cross split |
| tabular | unit conversion, missingness, categorical vocabularies, scaling | scalers or imputers fit on full data |

## Output Contract

Preprocessing should produce a model-ready object with explicit validity:

$$
z_i =
(a_i, e_i, m_i, q_i)
$$

where $a_i$ is the tensor or structured input, $e_i$ is the entity mapping back to the raw record, $m_i$ is metadata needed for labels and splits, and $q_i$ records quality flags such as missing atoms, invalid molecules, truncated sequence, or filtered fields.

For graph, coordinate, and sequence inputs, record:

- shape convention and padding or masking rule;
- node, edge, residue, atom, token, or region order;
- unit and coordinate frame for geometric data;
- mapping from processed indices back to original entities;
- invalid-example policy: reject, repair, mask, or keep with a flag.

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

## Public Note Checklist

- Does the note state whether preprocessing is deterministic or learned?
- Does it state which parameters are fit on training data only?
- Does it distinguish raw entity identity from processed representation identity?
- Does it preserve enough metadata to interpret labels, metrics, and failure cases?
- Does it avoid private paths, hostnames, credentials, and internal dataset names?

## Related

- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/data/data-schema|Data schema]]
- [[concepts/data/data-versioning|Data versioning]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
