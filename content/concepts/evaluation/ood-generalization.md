---
title: OOD Generalization
tags:
  - evaluation
  - methodology
  - generalization
---

# OOD Generalization

Out-of-distribution (OOD) generalization measures how a model performs on data that differs systematically from training — new chemotypes, protein families, time periods, or domains. In-distribution accuracy routinely overstates real-world performance because deployment is rarely IID.

The OOD gap compares risk under different distributions:

$$
\Delta_{\mathrm{OOD}}
= R_{p_{\mathrm{ood}}}(f)
- R_{p_{\mathrm{iid}}}(f)
$$

The shift must be named; otherwise OOD becomes a vague label.

## Shift Types

| Shift | Unit That Changes | Examples | Common Split |
| --- | --- | --- | --- |
| Covariate shift | $p(x)$ | image domain, sequence length, molecule size | source or time split |
| Label shift | $p(y)$ | class prevalence, active ratio, disease subtype mix | stratified source split |
| Conditional shift | $p(y\mid x)$ | assay protocol change, species difference | assay/source split |
| Support shift | available region of $x$ | novel scaffold, unseen protein family | scaffold or family split |
| Temporal shift | collection time | newer database releases, post-training papers | time split |
| Structural shift | object geometry or preparation | predicted vs experimental structures | structure-source split |

The strongest OOD claim usually names both the changed unit and the split rule.

## Risk Form

IID validation estimates:

$$
\hat{R}_{\mathrm{iid}}
=
\frac{1}{n_{\mathrm{iid}}}
\sum_{i=1}^{n_{\mathrm{iid}}}
\mathcal{L}(f(x_i),y_i)
$$

OOD evaluation estimates:

$$
\hat{R}_{\mathrm{ood}}
=
\frac{1}{n_{\mathrm{ood}}}
\sum_{i=1}^{n_{\mathrm{ood}}}
\mathcal{L}(f(x_i),y_i)
$$

The important evidence is not only $\hat{R}_{\mathrm{ood}}$, but the gap, uncertainty, and whether the OOD set actually represents the claimed deployment shift.

## Practical Checks

- Define the shift explicitly: structural, temporal, source, or covariate.
- Use grouped/structured splits (scaffold, family, time) to simulate the shift.
- Report the IID-to-OOD gap, not just one number.
- Check calibration under shift — confidence degrades before accuracy does.
- State the applicability domain: where the model is and is not expected to hold.
- Confirm preprocessing and model selection did not use OOD labels.
- Report enough sample counts per OOD slice to make the estimate meaningful.

## Computational Biology Route

| Claim | OOD Unit | Better Evidence |
| --- | --- | --- |
| new chemotypes | molecular scaffold | scaffold split plus activity-cliff analysis |
| new targets | protein or target family | protein-family split plus target-level metrics |
| new complexes | protein-ligand pair | pair split controlling both ligand and protein overlap |
| new assays | assay/source context | assay harmonization and source-held-out split |
| new structures | structure source/preparation | experimental vs predicted/template policy |
| new genome regions | region/chromosome/time | region-held-out or time-held-out split |

## Claim Template

An OOD result should name:

- Source distribution.
- Shifted target distribution.
- Split rule used to simulate the shift.
- Primary metric and uncertainty interval.
- Failure cases and applicability boundary.

Without these pieces, "OOD" is only a label, not an evaluation claim.

## Anti-Patterns

| Anti-Pattern | Why It Fails |
| --- | --- |
| Calling a random split OOD | row-level randomness usually preserves near duplicates |
| Reporting only one OOD number | no comparison to IID or baseline gap |
| Tuning on the OOD test set | turns OOD into validation |
| Ignoring label shift | metric changes may reflect prevalence, not model quality |
| Using broad "generalizes" wording | hides which deployment region is supported |

## Related

- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/data/dataset-shift|Dataset shift]]
- [[concepts/data/data-distribution|Data distribution]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/robustness|Robustness]]
- [[concepts/evaluation/index|Evaluation]]
