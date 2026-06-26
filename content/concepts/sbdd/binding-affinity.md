---
title: Binding Affinity
tags:
  - sbdd
  - docking
  - evaluation
---

# Binding Affinity

Binding affinity measures how strongly a ligand binds to a target. It is related to thermodynamics, but ML datasets often provide assay-dependent proxies such as $K_d$, $K_i$, $\mathrm{IC}_{50}$, or activity labels.

A common thermodynamic relation is:

$$
\Delta G
= RT\ln K_d
$$

where $\Delta G$ is binding free energy, $R$ is the gas constant, $T$ is temperature, and $K_d$ is the dissociation constant.

Many datasets use transformed values:

$$
pK_d = -\log_{10}(K_d)
$$

Higher $pK_d$ means stronger binding when the units are consistent.

Affinity values should be interpreted with the [[entities/target-assay-label|Target-assay-label contract]] because endpoint, assay, unit, target construct, and censoring change what the label means.

## Checks

- Is the target $K_d$, $K_i$, $\mathrm{IC}_{50}$, enrichment, or a binary activity label?
- Are units, assay conditions, and censoring handled explicitly?
- Is the target-assay-label contract preserved from raw measurement to benchmark label?
- Is the model predicting affinity, pose quality, or both?
- Are ligand scaffolds and protein families separated across splits?

## Related

- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/evaluation/metric|Metric]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[entities/assay|Assay]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
