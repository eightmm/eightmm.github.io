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

## Related

- [[entities/dataset|Dataset]]
- [[concepts/data/data-versioning|Data versioning]]
- [[concepts/data/missing-data|Missing data]]
- [[concepts/data/censored-label|Censored label]]
- [[concepts/data/weak-label|Weak label]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/evaluation/leakage|Leakage]]
