---
title: Dataset Construction Checklist
tags:
  - data
  - evaluation
  - methodology
---

# Dataset Construction Checklist

A dataset construction checklist turns a modeling idea into an auditable dataset. It should be written before interpreting metrics, because most evaluation failures start as data-definition failures.

A compact construction pipeline is:

$$
\mathcal{R}
\xrightarrow{\text{curation}}
\mathcal{D}_{\mathrm{raw}}
\xrightarrow{P_\phi}
\mathcal{D}_{\mathrm{model}}
\xrightarrow{s}
(\mathcal{D}_{\mathrm{train}},\mathcal{D}_{\mathrm{val}},\mathcal{D}_{\mathrm{test}})
$$

where $\mathcal{R}$ is the source record set, $P_\phi$ is the preprocessing contract, and $s$ is the split rule.

## Checklist

- Define the [[concepts/data/example-unit|example unit]].
- Define the [[concepts/data/split-unit|split unit]].
- Write the [[concepts/data/data-schema|data schema]].
- State [[concepts/data/label-semantics|label semantics]] and missing-label policy.
- Record source, license, collection process, and [[concepts/data/metadata-provenance|metadata provenance]].
- Write the [[concepts/data/preprocessing-contract|preprocessing contract]].
- Decide the [[concepts/data/sampling-strategy|sampling strategy]].
- Check duplicates, near-duplicates, and grouped leakage.
- Build a [[concepts/data/dataset-card|dataset card]].
- Attach an [[concepts/evaluation/evaluation-protocol|evaluation protocol]].

## Bio-AI Checks

- Molecules: standardization, tautomer/protonation/stereo policy, scaffold grouping.
- Proteins: sequence identity, family grouping, structure source, confidence, missing regions.
- Protein-ligand complexes: receptor preparation, ligand preparation, pocket definition, pose source.
- Assays: target context, endpoint, threshold, units, censoring, and harmonization.
- Structure-based tasks: template leakage, ligand-defined pocket leakage, and deployment-available context.

## Public Writing Boundary

For a public blog, describe dataset construction as a general method unless the dataset is public and the metadata is verified. Do not publish private dataset names, collaborator details, internal paths, unpublished results, or non-public task identifiers.

## Related

- [[concepts/data/index|Data]]
- [[concepts/data/dataset-card|Dataset card]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
