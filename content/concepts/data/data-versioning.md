---
title: Data Versioning
tags:
  - data
  - reproducibility
  - provenance
---

# Data Versioning

Data versioning records exactly which data snapshot, filters, preprocessing steps, and labels were used for a run or result. Without it, model comparisons can silently compare different datasets.

[[concepts/data/data-lineage|Data lineage]] explains how that version was produced; data versioning gives it a stable identifier.

A dataset version can be described as:

$$
v_{\mathcal{D}}
=
H(\text{sources}, \text{filters}, \text{labels}, \text{splits}, \text{preprocessing})
$$

where $H$ is a content hash or another stable identifier.

## Key Ideas

- Version the raw source, processed dataset, split files, and label definitions.
- A changed filter or deduplication rule can invalidate previous metrics.
- Dataset versions should be linked to training runs, paper notes, and public claims.
- Provenance should explain where data came from and what transformations were applied.

## Practical Checks

- Can the exact train/validation/test split be reconstructed?
- Are labels and metadata versioned with the examples?
- Are derived files traceable to raw sources and scripts?
- Does the run log record dataset version identifiers?
- Are private or restricted sources excluded from public notes?

## Related

- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/data/data-lineage|Data lineage]]
- [[concepts/data/data-curation|Data curation]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/systems/training-run|Training run]]
- [[logs/public-log-format|Public log format]]
