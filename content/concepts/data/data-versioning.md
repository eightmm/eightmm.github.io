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

Data versioning is not only a file checksum. A useful version binds data, labels, splits, and preprocessing:

$$
V
=
(S,F,L,P,\Pi)
$$

where $S$ is source records, $F$ is filtering and deduplication, $L$ is label construction, $P$ is preprocessing, and $\Pi$ is the split policy.

## Key Ideas

- Version the raw source, processed dataset, split files, and label definitions.
- A changed filter or deduplication rule can invalidate previous metrics.
- Dataset versions should be linked to training runs, paper notes, and public claims.
- Provenance should explain where data came from and what transformations were applied.

## Version Boundary

| Change | New Version? | Why |
| --- | --- | --- |
| raw source changed | yes | example population changed |
| deduplication or filtering changed | yes | train/test composition changed |
| label threshold or unit changed | yes | target semantics changed |
| train/validation/test split changed | yes | evaluation claim changed |
| deterministic serialization changed only | maybe | content may be identical |
| cached embedding recomputed with same model and data | record artifact version | representation artifact changed |

## Practical Checks

- Can the exact train/validation/test split be reconstructed?
- Are labels and metadata versioned with the examples?
- Are derived files traceable to raw sources and scripts?
- Does the run log record dataset version identifiers?
- Are private or restricted sources excluded from public notes?
- Does the paper or project claim state which dataset version supports the result?
- Can old metrics be invalidated when the dataset version changes?

## Related

- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/data/data-lineage|Data lineage]]
- [[concepts/data/data-curation|Data curation]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/systems/training-run|Training run]]
- [[logs/public-log-format|Public log format]]
