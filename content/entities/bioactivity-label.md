---
title: Bioactivity Label
tags:
  - entities
  - molecular-modeling
  - labels
---

# Bioactivity Label

A bioactivity label is a measured or curated value that describes how a molecule, ligand, protein, target, assay, or context behaves in a biological or chemical setup.

It is not just a number. A label should be read as:

$$
y
=
\operatorname{measure}(m, t, a, c)
$$

where $m$ is a molecule or ligand, $t$ is a target, $a$ is an assay protocol, and $c$ is measurement context such as endpoint, organism, construct, dose, time, unit, and preprocessing.

This is the label side of the [[entities/target-assay-label|Target-assay-label contract]].

## Common Forms

- Continuous: $K_d$, $K_i$, $\mathrm{IC}_{50}$, $\mathrm{EC}_{50}$, percent inhibition, activity score, affinity proxy.
- Transformed: $pK_d$, $pK_i$, $pIC_{50}$, log concentration, standardized score.
- Binary: active/inactive after a threshold.
- [[concepts/data/censored-label|Censored]]: values reported as greater-than or less-than limits.
- [[concepts/data/weak-label|Weak]] or [[concepts/data/missing-data|missing]]: untested, inferred, aggregated, or source-dependent labels.

## Why It Matters

- The same molecule-target pair can have different labels under different assay contexts.
- A binary active label may hide endpoint type, threshold, unit, and censoring.
- Regression, classification, ranking, and virtual screening can use the same raw records differently.
- Negative examples may mean measured inactive, not selected, or simply unobserved.

## Label Semantics Contract

| Field | Question | Example |
| --- | --- | --- |
| entity unit | what object receives the label? | molecule, molecule-target pair, complex, assay row |
| target | is the label target-conditioned? | kinase target, receptor, protein family |
| assay | which protocol produced it? | binding, functional, cell viability |
| endpoint | what quantity is measured? | $K_d$, $\mathrm{IC}_{50}$, percent inhibition |
| transformation | raw or transformed? | $pIC_{50}=-\log_{10}(IC_{50})$ |
| threshold | how binary labels are made | active if $pIC_{50} \ge \tau$ |
| censoring | inequality or detection limit | $>10\ \mu M$, $<1\ nM$ |
| replicate policy | aggregation rule | median, mean, best, latest, source-specific |
| negative meaning | inactive, decoy, unobserved, or assumed negative | affects ranking and screening |

For concentration-style labels:

$$
pIC_{50}
=
-\log_{10}\left(IC_{50}\,[\mathrm{M}]\right)
$$

Unit conversion must happen before the transform.

## Task Rewrites

The same raw bioactivity table can define different tasks:

| Task | Label Interpretation |
| --- | --- |
| regression | predict continuous endpoint after unit/transform policy |
| classification | predict thresholded active/inactive label |
| ranking | rank compounds for one target or assay context |
| multitask learning | predict assay-specific labels with missing entries |
| virtual screening | enrich measured actives in a candidate pool |
| target-conditioned prediction | model $p(y\mid m,t,a)$ rather than $p(y\mid m)$ |

The paper should state which rewrite it uses.

## Leakage Risks

| Leakage | Why It Happens |
| --- | --- |
| molecule duplicate | same compound appears under different salts or strings |
| target overlap | homologous targets split across train/test |
| assay/source overlap | assay identity acts as a shortcut |
| threshold tuning | binary threshold chosen after seeing test behavior |
| replicate aggregation | test information leaks into global aggregation |
| negative construction | decoys encode benchmark artifacts |

## Checks

- What endpoint does the label represent?
- What unit, transformation, threshold, and censoring policy are used?
- Is the label molecule-only, target-conditioned, assay-conditioned, or complex-conditioned?
- Are replicate measurements aggregated, kept separate, or filtered?
- Are conflicting labels resolved globally or kept assay-specific?
- Does the split prevent molecule, target, assay, or source leakage?
- Is the label valid for the intended task, or only for the original assay context?
- Are unmeasured pairs treated differently from measured inactive pairs?
- Are labels aggregated before or after train/test splitting?

## Related

- [[entities/assay|Assay]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[entities/molecule|Molecule]]
- [[entities/ligand|Ligand]]
- [[entities/target|Target]]
- [[entities/dataset|Dataset]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/missing-data|Missing data]]
- [[concepts/data/censored-label|Censored label]]
- [[concepts/data/weak-label|Weak label]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/molecular-modeling/molecular-property-prediction|Molecular property prediction]]
