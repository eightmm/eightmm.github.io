---
title: Data Curation
tags:
  - data
  - dataset
  - curation
---

# Data Curation

Data curation turns raw records into a dataset that can support a clear modeling claim. It includes filtering, deduplication, standardization, label cleanup, metadata checks, and split design.

The curated dataset is a filtered subset of raw records:

$$
\mathcal{D}_{\mathrm{curated}}
= \{r \in \mathcal{D}_{\mathrm{raw}} : C(r)=1\}
$$

where $C(r)$ is the curation criterion.

## Common Steps

- Define the unit of prediction: molecule, protein, complex, image, document, clip, assay row, or trajectory.
- Standardize representations before deduplication.
- Remove invalid, ambiguous, or out-of-scope records.
- Preserve metadata needed for auditing.
- Assign splits after deduplication and group construction.

## Checks

- Is the filtering rule written down?
- Are invalid records removed or marked?
- Are duplicates exact, canonicalized, or near-duplicate?
- Are labels comparable after merging sources?
- Does curation use any test-set-specific information?

## Related

- [[entities/dataset|Dataset]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
