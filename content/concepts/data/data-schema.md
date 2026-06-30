---
title: Data Schema
tags:
  - data
  - schema
---

# Data Schema

A data schema defines the fields, types, units, allowed values, and relationships that make a dataset interpretable. Without a schema, preprocessing and evaluation can silently change the task.

A simple dataset can be viewed as rows and typed fields:

$$
\mathcal{D}
=
\{r_i\}_{i=1}^{n},
\qquad
r_i
=
\{c_j: v_{ij}\}_{j=1}^{d}
$$

where $c_j$ is a column or field name and $v_{ij}$ is the value for row $i$.

## Schema Elements

- Field name and semantic meaning.
- Data type: numeric, categorical, ordinal, text, sequence, graph, coordinate, timestamp, or identifier.
- Unit and scale.
- Missing-value policy.
- Censoring, weak-label, and unknown-label policy.
- Allowed value range or vocabulary.
- Entity relationships and keys.
- Version and source.

## Schema Contract

A useful schema distinguishes identity, input, label, metadata, and split fields.

$$
r_i
=
(e_i,\ x_i,\ y_i,\ m_i,\ s_i)
$$

| Field class | Meaning | Common risk |
| --- | --- | --- |
| entity key $e_i$ | molecule, protein, document, user, assay, structure id | leaks duplicates or grouping |
| input $x_i$ | features available to the model | includes post-label information |
| label $y_i$ | prediction target | unit/censoring/threshold ambiguity |
| metadata $m_i$ | source, batch, time, protocol | confounding or leakage |
| split unit $s_i$ | grouping for evaluation | row split hides related examples |

The schema should say which fields are allowed at inference time. A field can be valid metadata but invalid model input.

## Type and Unit Examples

| Data kind | Schema detail to record |
| --- | --- |
| molecule | identifier type, salt/protonation handling, canonicalization policy |
| protein sequence | alphabet, unknown residue policy, isoform or fragment boundary |
| structure | coordinate frame, missing residues/atoms, resolution or prediction source |
| assay label | unit, transform, censoring, target, protocol |
| text/document | source, timestamp, section, chunking policy |
| image/video | resolution, color/channel convention, frame sampling |

## Inference Availability

For each field, ask whether it exists before prediction.

| Field | Use as input? | Reason |
| --- | --- | --- |
| raw measurement used to derive label | no | target leakage |
| assay/source batch | maybe | can improve in-domain but harm transfer claim |
| molecule id | usually no | memorization risk |
| public descriptor computable from input | yes | reproducible at inference |
| split group | no | evaluation metadata only |

## Schema Versioning

Schema changes can change the task even when file names stay the same.

| Change | Why it matters |
| --- | --- |
| unit conversion | target scale changes |
| threshold change | class boundary changes |
| missing-value encoding | model may learn missingness artifact |
| canonicalization update | duplicates and splits can change |
| label harmonization | comparable assays may be merged or separated |

## Why It Matters

- Units can change the target without changing the column name.
- Identifiers can leak split, batch, source, target, or label information.
- Categorical vocabularies can drift between train and deployment.
- Nested objects such as molecules, sequences, structures, and assays require schema-level relationships.

## Checks

- What is one row or example?
- Which fields are identifiers rather than predictive signals?
- Which fields are available at inference time?
- Are units, transforms, and censoring rules explicit?
- Are missing labels and weak labels encoded separately from true negatives?
- Can another person reconstruct the same model-ready table?
- Are identifiers and split fields excluded from model inputs unless explicitly justified?
- Is the schema version tied to the dataset version and preprocessing contract?

## Related

- [[entities/dataset|Dataset]]
- [[concepts/data/data-versioning|Data versioning]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/missing-data|Missing data]]
- [[concepts/data/censored-label|Censored label]]
- [[concepts/data/weak-label|Weak label]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/evaluation/leakage|Leakage]]
