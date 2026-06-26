---
title: Metadata and Provenance
tags:
  - data
  - metadata
  - provenance
---

# Metadata and Provenance

Metadata records the context around an example. Provenance records where an example came from and how it was transformed. Both are required for reproducible datasets and leakage audits.

When provenance is chained across source, curation, preprocessing, and splits, it becomes [[concepts/data/data-lineage|data lineage]].

A dataset item should be treated as:

$$
r_i = (x_i, y_i, m_i, s_i)
$$

where $m_i$ is metadata and $s_i$ is source or provenance information.

## Useful Metadata

- Source dataset or publication.
- Collection time or version.
- Assay, protocol, organism, target, or experimental context.
- Preprocessing and filtering flags.
- Group IDs for split construction.
- License and citation information.

## Provenance Tuple

For each example, provenance should be enough to reconstruct how it entered the dataset:

$$
s_i
=
(\text{source},
\text{source version},
\text{record id},
\text{curation step},
\text{preprocessing step},
\text{split assignment})
$$

This does not mean every field is model input. Some fields are audit-only and should be excluded from training to avoid leakage.

## Metadata Roles

Metadata has different roles:

$$
m_i
\in
\{\text{input feature},
\text{label context},
\text{split key},
\text{audit field},
\text{citation field}\}
$$

The same column should not silently change roles. For example, a source identifier may be necessary for auditing but harmful as an input feature if it encodes the label distribution.

## Failure Modes

- Source fields are stripped, making later leakage checks impossible.
- Metadata is used as model input even though it is unavailable at deployment.
- A split key is computed from private or post-outcome information.
- Versions are missing, so refreshed sources silently change examples.
- Public and private sources are merged without clear boundaries.

## Checks

- Can the record be traced back to its source?
- Is metadata used as input, filtering context, or audit-only information?
- Does metadata leak the label or test membership?
- Are dataset versions stable and cited?
- Are public and private sources separated clearly?
- Is each metadata field assigned a role?
- Is source/version information preserved after preprocessing?
- Are deployment-unavailable fields excluded from model input?

## Related

- [[entities/dataset|Dataset]]
- [[concepts/data/data-lineage|Data lineage]]
- [[concepts/data/data-curation|Data curation]]
- [[concepts/data/data-versioning|Data versioning]]
- [[concepts/data/dataset-card|Dataset card]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[papers/workflows/paper-note-format|Paper note format]]
