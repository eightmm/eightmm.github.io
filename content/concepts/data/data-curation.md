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

## Curation Contract

A useful curation step should be reproducible:

$$
\mathcal{D}_{\mathrm{curated}}
=
\operatorname{curate}
(\mathcal{D}_{\mathrm{raw}};\ C,\ S,\ H)
$$

where $C$ is the inclusion/exclusion rule, $S$ is the standardization protocol, and $H$ is the hashing or identity policy used for deduplication and split grouping.

The contract should state:

- what is one record;
- what is one modeled example;
- what is removed, repaired, or marked invalid;
- what metadata is preserved;
- what identity key defines duplicates or near-duplicates.

## Order Matters

For many datasets, the order should be:

$$
\text{raw}
\rightarrow
\text{standardize}
\rightarrow
\text{deduplicate}
\rightarrow
\text{group}
\rightarrow
\text{split}
\rightarrow
\text{fit preprocessing}
$$

Splitting before standardization or deduplication can leak near-duplicates across train and test.

## Failure Modes

- Filtering rules are changed after seeing validation/test metrics.
- Invalid records are silently dropped without counts.
- Label cleanup merges incomparable measurements.
- Metadata needed for leakage audits is removed too early.
- Split keys are based on row IDs instead of the real generalization unit.

## Checks

- Is the filtering rule written down?
- Are invalid records removed or marked?
- Are duplicates exact, canonicalized, or near-duplicate?
- Are labels comparable after merging sources?
- Does curation use any test-set-specific information?
- Are before/after record counts reported by removal reason?
- Is the split created after standardization and deduplication?
- Can another person reproduce the same curated dataset from the raw sources?

## Related

- [[entities/dataset|Dataset]]
- [[concepts/data/data-schema|Data schema]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/data/data-versioning|Data versioning]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
