---
title: Metadata and Provenance
tags:
  - data
  - metadata
  - provenance
---

# Metadata and Provenance

Metadata records the context around an example. Provenance records where an example came from and how it was transformed. Both are required for reproducible datasets and leakage audits.

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

## Checks

- Can the record be traced back to its source?
- Is metadata used as input, filtering context, or audit-only information?
- Does metadata leak the label or test membership?
- Are dataset versions stable and cited?
- Are public and private sources separated clearly?

## Related

- [[entities/dataset|Dataset]]
- [[concepts/data/data-curation|Data curation]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[papers/paper-note-format|Paper note format]]
