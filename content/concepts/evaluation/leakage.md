---
title: Leakage
tags:
  - evaluation
  - methodology
  - data
---

# Leakage

Leakage is when information from the test set reaches the model during training, inflating reported performance that then collapses in real use. It hides in duplicate records, shared identifiers across splits, target-derived features, and preprocessing fit on the full dataset.

The split should satisfy an independence assumption:

$$
\mathcal{D}_{\mathrm{train}}
\cap_{\mathrm{information}}
\mathcal{D}_{\mathrm{test}}
= \varnothing
$$

The subscript means exact row overlap is not enough; near-duplicates, shared groups, and target-derived features can also transmit information.

## Practical Checks

- Split before any fitting — scalers, encoders, and imputers learn on train only.
- Deduplicate and check for near-duplicates across train/val/test.
- Ensure grouped entities (molecule, protein, patient) never span splits.
- Audit features for any that encode the label or future information.
- If a result looks too good, suspect leakage before celebrating.

## Common Leakage Channels

- Duplicate or near-duplicate examples across splits.
- Shared scaffold, protein family, subject, source, or time period across splits.
- Preprocessing fit on the full dataset before splitting.
- Feature selection, imputation, normalization, or vocabulary building that sees test data.
- Target-derived features, post-outcome metadata, or future information.
- Test-set iteration during model selection.

## Domain-Specific Channels

| Domain | Leakage unit | Example |
|---|---|---|
| molecule property | duplicate, salt form, tautomer, scaffold, close analog | same chemotype appears in train and test |
| protein modeling | sequence identity cluster, domain, template, MSA database | homolog or template crosses the boundary |
| protein-ligand task | protein, ligand scaffold, pair, assay source | target novelty confused with ligand interpolation |
| structure prediction | template, predicted structure source, database date | benchmark structure appears in retrieval context |
| LLM or agent task | prompt family, answer key, tool trace, benchmark discussion | final task appears in pretraining or prompt examples |
| clinical or assay data | subject, lab, batch, time, source | source-specific artifacts predict labels |

## Leakage Test

For a split function $s$ and information unit $u$, a basic grouped-leakage condition is:

$$
s(x_i)\ne s(x_j)
\Rightarrow
u(x_i)\ne u(x_j)
$$

This condition must be checked for every unit that could transmit target-relevant information.

Equivalently, no leakage-relevant group should appear in more than one split:

$$
\left|
\{s(x): u(x)=a\}
\right|
=
1
\quad
\forall a
$$

## Audit Evidence

| Audit | Evidence to record |
|---|---|
| exact duplicate check | duplicate count before and after split |
| near-duplicate check | similarity threshold and cross-split hits |
| grouped split audit | number of groups per split and cross-split violations |
| preprocessing audit | which statistics were fit on train only |
| label/source audit | source distribution and label distribution per split |
| test-iteration audit | number of final-test inspections or submissions |

An audit should report both the rule and the residual risk. "No leakage" is too broad unless the leakage unit is named.

## Related

- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/data/data-schema|Data schema]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/test-set-contamination|Test-set contamination]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/index|Evaluation]]
