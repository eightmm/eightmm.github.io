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

## Harmonization Fields

| Field | Examples | Why It Matters |
| --- | --- | --- |
| target identity | protein, isoform, species, construct, mutation | different constructs can change activity |
| endpoint | $K_d$, $K_i$, $\mathrm{IC}_{50}$, percent inhibition, binary active | endpoints are not interchangeable labels |
| unit and transform | nM, uM, pIC50, Delta G, normalized score | transformations change loss scale and thresholds |
| censoring | `>`, `<`, detection limit, inactive above cutoff | censoring is not exact measurement |
| replicate policy | mean, median, best, latest, source-prioritized | aggregation changes label noise |
| assay source | lab, campaign, dataset, protocol, publication | source can leak into train/test splits |

## Common Conversions

When a concentration $c$ is converted to a logarithmic activity score:

$$
\mathrm{p}c = -\log_{10}\left(\frac{c}{1\ \mathrm{M}}\right)
$$

This conversion is only meaningful when $c$ is a comparable endpoint and unit. Mixing $\mathrm{IC}_{50}$, $K_i$, and $K_d$ can be useful for rough triage but should not be presented as a clean physical target without stating the approximation.

## Split Implications

| Claim | Split or Control |
| --- | --- |
| new molecule in same assay setting | scaffold split with assay held fixed |
| new assay or campaign | source/campaign holdout |
| new target family | protein-family split plus assay-source control |
| cross-dataset transfer | external dataset or source holdout with harmonization audit |
| target-conditioned activity | keep target, assay, endpoint, unit, and threshold explicit |

## Checks

- Are endpoint types mixed, such as $K_d$, $K_i$, $\mathrm{IC}_{50}$, inhibition percent, or binary labels?
- Are units and transformations recorded?
- Are censored values such as `>` or `<` handled explicitly?
- Are replicate measurements aggregated deliberately?
- Are conflicting molecule-target labels resolved or kept assay-specific?
- Does the split hold out assays or campaigns when that is the deployment claim?
- Is the label direction consistent, so larger always means more active or less active?
- Are source-specific missingness and measurement limits documented?

## Related

- [[entities/assay|Assay]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[entities/dataset|Dataset]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
