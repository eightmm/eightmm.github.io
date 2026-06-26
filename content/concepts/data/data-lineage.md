---
title: Data Lineage
tags:
  - data
  - provenance
  - reproducibility
---

# Data Lineage

Data lineage records how a dataset was derived from source records through curation, filtering, preprocessing, labeling, splitting, and evaluation. It connects [[concepts/data/metadata-provenance|Metadata and provenance]], [[concepts/data/data-versioning|Data versioning]], [[concepts/data/preprocessing-contract|Preprocessing contract]], and [[concepts/evaluation/evaluation-protocol|Evaluation protocol]].

A compact lineage graph is:

$$
\mathcal{S}
\xrightarrow{\text{curation}}
\mathcal{D}_{\mathrm{raw}}
\xrightarrow{\text{filter}}
\mathcal{D}_{\mathrm{clean}}
\xrightarrow{P_\phi}
\mathcal{D}_{\mathrm{model}}
\xrightarrow{s}
(\mathcal{D}_{\mathrm{train}}, \mathcal{D}_{\mathrm{val}}, \mathcal{D}_{\mathrm{test}})
$$

where $\mathcal{S}$ is the source record set, $P_\phi$ is the preprocessing contract, and $s$ is the split rule.

For a run, lineage should identify:

$$
L_{\mathcal{D}}
=
(\text{source}, \text{curation}, \text{filters}, \text{labels}, \text{preprocessing}, \text{splits}, \text{version})
$$

## Why It Matters

- A small filtering change can change benchmark difficulty.
- Split leakage often comes from lineage mistakes, not model code.
- Reproducibility requires tracing metrics back to a dataset version and split.
- Public claims should only depend on public or safely summarized sources.
- Dataset lineage separates data changes from architecture or optimizer changes.

## Lineage Record

A useful lineage record should include:

- Public source name, citation, license, and retrieval date or version.
- Raw record schema and stable identifiers.
- Inclusion and exclusion filters.
- Deduplication and near-duplicate policy.
- Label generation, harmonization, censoring, and missing-label policy.
- Preprocessing, tokenization, standardization, featurization, and normalization contracts.
- Split rule, split unit, random seed, group IDs, and leakage checks.
- Dataset version identifier and any content hashes that can be shared publicly.
- Link to the training run, evaluation protocol, and dataset card.

## Bio Lineage

For structure-based and molecular datasets, lineage should preserve entity relationships:

$$
(\text{molecule}, \text{target}, \text{assay}, \text{label}, \text{unit}, \text{source})
$$

or, for structure data:

$$
(\text{protein}, \text{ligand}, \text{complex}, \text{pose}, \text{pocket}, \text{source})
$$

Flattening these records too early can hide assay changes, target ambiguity, template leakage, or pocket-definition leakage.

## Checks

- Can every processed example be traced to a public or safely summarized source?
- Are filters and deduplication rules recorded before metric interpretation?
- Is the split generated from the intended split unit rather than row order?
- Are labels tied to source, unit, threshold, and censoring context?
- Can a run be associated with one dataset lineage identifier?
- Are private paths, private dataset names, collaborator details, and unpublished results excluded from public notes?

## Related

- [[concepts/data/index|Data]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/data/data-versioning|Data versioning]]
- [[concepts/data/dataset-card|Dataset card]]
- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[entities/dataset|Dataset]]
