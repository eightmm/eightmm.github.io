---
title: Assay Harmonization
tags:
  - evaluation
  - assay
  - dataset
---

# Assay Harmonization

Assay harmonization is the process of making measurement records comparable before training or evaluating a model. In chem-bio ML, labels with the same column name can come from different protocols, targets, units, and biological contexts.

A measurement should be treated as:

$$
y
=
\operatorname{measure}
(M, T, A, c)
$$

where $M$ is a molecule, $T$ is a target, $A$ is an assay protocol, and $c$ contains conditions such as organism, construct, endpoint, units, and environment.

## Checks

- Are endpoint types mixed, such as $K_d$, $K_i$, $\mathrm{IC}_{50}$, inhibition percent, or binary labels?
- Are units and transformations recorded?
- Are censored values such as `>` or `<` handled explicitly?
- Are replicate measurements aggregated deliberately?
- Are conflicting molecule-target labels resolved or kept assay-specific?
- Does the split hold out assays or campaigns when that is the deployment claim?

## Related

- [[entities/assay|Assay]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[entities/dataset|Dataset]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
