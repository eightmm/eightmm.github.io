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

## Endpoint Boundary

| Endpoint | Meaning | Caveat |
| --- | --- | --- |
| $K_d$ | equilibrium dissociation constant | closest to direct binding equilibrium when measured cleanly |
| $K_i$ | inhibition constant | depends on assay model and competitor assumptions |
| $\mathrm{IC}_{50}$ | concentration for half-maximal inhibition | depends on assay conditions, substrate, time, and target context |
| activity class | thresholded active/inactive label | threshold, source, and missing negatives dominate semantics |
| enrichment | screening recovery of known actives | ranking utility, not a thermodynamic affinity value |

Do not treat these endpoints as interchangeable without an explicit conversion or harmonization rule.

## Transformation and Units

If $K_d$ is measured in molar units:

$$
\Delta G^\circ
=
RT\ln K_d
$$

and

$$
pK_d
=
-\log_{10}\left(\frac{K_d}{1\ \mathrm{M}}\right).
$$

For $K_d < 1\,\mathrm{M}$, larger $pK_d$ usually means stronger binding and more negative $\Delta G^\circ$. A note should state units before applying $pK_d$, $pK_i$, or $pIC_{50}$ transformations.

## Modeling Boundary

Affinity prediction is usually:

$$
\hat{a}
=
f_\theta(P, L, c)
$$

where $P$ is the protein or target context, $L$ is the ligand, and $c$ is assay or structural context. If $c$ differs across train and test, the model may learn assay/source effects rather than binding.

Affinity is separate from pose quality:

$$
\text{low RMSD}
\not\Rightarrow
\text{accurate affinity}
$$

A docking workflow can produce a plausible pose and still fail to rank ligands by binding strength.

## Checks

- Is the target $K_d$, $K_i$, $\mathrm{IC}_{50}$, enrichment, or a binary activity label?
- Are units, assay conditions, and censoring handled explicitly?
- Is the target-assay-label contract preserved from raw measurement to benchmark label?
- Is the model predicting affinity, pose quality, or both?
- Are ligand scaffolds and protein families separated across splits?
- Is target or assay source leakage possible?
- Are censored values such as `>`, `<`, or thresholded actives handled as bounds rather than exact labels?
- Does the reported metric match the decision: numeric accuracy, ranking, enrichment, or calibration?

## Related

- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[entities/assay|Assay]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
