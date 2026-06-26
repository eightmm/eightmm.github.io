---
title: Dataset Split Contract
tags:
  - data
  - evaluation
  - leakage
---

# Dataset Split Contract

A dataset split contract records how examples are assigned to train, validation, and test sets. It makes the generalization claim reproducible and auditable.

A split function is:

$$
s : \mathcal{D} \rightarrow \{\mathrm{train}, \mathrm{val}, \mathrm{test}\}
$$

For a grouping function $g(x_i)$ that maps an example to its leakage-relevant unit, the contract should satisfy:

$$
g(x_i)=g(x_j)
\Rightarrow
s(x_i)=s(x_j)
$$

This prevents examples from the same split unit from crossing train, validation, and test.

## Required Fields

- Example unit: what one row or example means.
- Split unit: molecule, scaffold, protein family, subject, source, time block, document group, or another grouping key.
- Split rule: random, grouped, temporal, scaffold, protein-family, source-based, or domain holdout.
- Split sizes: counts for examples and split units in train, validation, and test.
- Deduplication rule: exact and near-duplicate handling before splitting.
- Preprocessing order: which steps happen before and after split assignment.
- Version: stable identifiers for raw data, processed data, and split files.
- Leakage audit: checks performed and known residual risks.

## Split Assignment Table

A public split contract can be summarized as:

| Field | Public value |
| --- | --- |
| Example unit | `to specify` |
| Split unit | `to specify` |
| Split rule | `to specify` |
| Validation role | model selection / early stopping / threshold tuning |
| Test role | final held-out estimate |
| Leakage checks | duplicates, grouped units, preprocessing, metadata |

Use `to specify` rather than inventing missing details.

## Molecular Modeling Examples

- Molecule property prediction: hold out molecular scaffolds or clusters.
- Protein representation: hold out sequence-identity clusters or protein families.
- Protein-ligand modeling: state whether the split holds out proteins, ligands, pairs, assays, or multiple axes.
- Structure-based tasks: document whether pockets, templates, poses, or receptor preparation use deployment-available information.

## Checks

- Does the split rule match the claim being made?
- Is validation the same type of shift as test, or only a tuning split?
- Are split files versioned and reconstructable?
- Are preprocessing statistics fit only on training data?
- Are public notes free of private dataset names, internal paths, unpublished metrics, and collaborator-specific details?

## Related

- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/dataset-card|Dataset card]]
- [[concepts/data/data-versioning|Data versioning]]
- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
