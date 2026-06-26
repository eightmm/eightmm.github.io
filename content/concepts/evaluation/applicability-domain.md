---
title: Applicability Domain
tags:
  - evaluation
  - generalization
  - molecular-modeling
---

# Applicability Domain

Applicability domain describes where a model's predictions are expected to be reliable. A high retrospective metric does not mean every new compound, protein, or structure is in-domain.

A simple nearest-neighbor domain score is:

$$
d_{\mathrm{AD}}(x)
= \min_{x_i\in \mathcal{D}_{\mathrm{train}}}
d(x,x_i)
$$

Predictions can then be stratified into in-domain and out-of-domain groups:

$$
\operatorname{inAD}(x)
= \mathbf{1}[d_{\mathrm{AD}}(x)\le \tau]
$$

Here $d$ is a domain-specific distance and $\tau$ is a validation-chosen threshold. The distance must match the object being modeled; a Tanimoto distance over molecules does not define protein novelty, and sequence identity does not define ligand novelty.

## Domain Axes

| Object | Common Distance | Typical Failure |
| --- | --- | --- |
| molecule | fingerprint Tanimoto, scaffold, physicochemical descriptor distance | analog series memorization |
| protein | sequence identity, family cluster, embedding distance, structural similarity | homolog leakage |
| protein-ligand complex | ligand distance plus target family distance plus pocket similarity | one side is novel but the other side is interpolated |
| generated structure | geometry validity, constraint distance, distributional coverage | high novelty with invalid samples |
| assay record | source, protocol, endpoint, unit, organism, construct | apparent generalization is assay-source memorization |

## Reporting Pattern

| Field | Required Detail |
| --- | --- |
| domain definition | object, representation, distance, threshold, and validation rule |
| threshold selection | validation-only selection; no test feedback |
| stratified metrics | in-domain, boundary, and out-of-domain subsets |
| uncertainty | confidence interval or paired comparison inside each subset |
| failure analysis | examples near the boundary and examples far outside training support |

## Checks

- What distance is used: fingerprint Tanimoto, sequence identity, embedding distance, or structural similarity?
- Is the threshold chosen on validation data, not test data?
- Are metrics reported separately for in-domain and out-of-domain examples?
- Does confidence degrade under distribution shift?
- Is calibration checked inside and outside the applicability domain?
- Does the paper confuse "novel ligand" with "novel protein-ligand interaction"?
- Is the applicability domain defined on the final input representation or only on raw metadata?

## Related

- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/evaluation/metric|Metric]]
