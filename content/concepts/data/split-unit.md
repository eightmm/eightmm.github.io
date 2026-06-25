---
title: Split Unit
tags:
  - data
  - evaluation
  - leakage
---

# Split Unit

A split unit is the entity that must stay on one side of a train, validation, or test split. It is often broader than one example.

If $g(i)$ maps example $i$ to a split unit, then a leakage-safe split requires:

$$
g(i) = g(j)
\Rightarrow
\operatorname{split}(i)
=
\operatorname{split}(j)
$$

Examples sharing the same split unit should not be separated across train and test.

## Examples

- Molecule task: scaffold, molecular cluster, or standardized molecule.
- Protein task: sequence identity cluster, protein family, or target.
- Structure-based task: protein family plus ligand scaffold or complex family.
- Assay task: assay, campaign, source, or target group.
- Document task: source document, author, time period, or collection.
- User-event task: user, session, device, or time block.

## Checks

- What entity defines the generalization claim?
- Is the split unit broader than the row identifier?
- Are near-duplicates grouped before splitting?
- Are preprocessing, scaling, and threshold choices fit only on train?
- Does the paper or benchmark report the split unit explicitly?

## Related

- [[concepts/data/example-unit|Example unit]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
