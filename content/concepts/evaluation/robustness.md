---
title: Robustness
tags:
  - evaluation
  - robustness
  - methodology
---

# Robustness

Robustness measures whether model behavior remains acceptable under perturbations, noise, missing inputs, distribution shifts, or implementation changes.

A robustness check compares performance under a perturbation family $\mathcal{T}$:

$$
R_{\mathcal{T}}(f)
=
\mathbb{E}_{(x,y)\sim p}
\left[
\mathcal{L}(f(T(x)),y)
\right],
\qquad T\sim\mathcal{T}
$$

The robustness gap is:

$$
\Delta_{\mathcal{T}}
= R_{\mathcal{T}}(f) - R(f)
$$

## Perturbation Types

- Input noise, crop, masking, corruption, or missing modality.
- Tokenization or formatting changes.
- Molecule standardization or conformer variation.
- Protein sequence homolog or structure-quality changes.
- Hardware, precision, batching, or decoding changes during inference.

## Robustness Contract

| Field | Question | Example |
| --- | --- | --- |
| Perturbation family | What changes between clean and perturbed input? | corruption, masking, conformer, sequence mutation |
| Label preservation | Should the correct label stay the same? | scaffold-preserving analog vs activity cliff |
| Severity | How large is the perturbation? | noise level, mutation count, RMSD range |
| Slice | Where is robustness measured? | class, scaffold, family, source, modality |
| Metric | What behavior is allowed to degrade? | accuracy, calibration, ranking, geometry validity |
| Failure policy | What counts as invalid output? | NaN, failed generation, impossible pose, abstention |

Robustness is only meaningful when the perturbation is plausible and the expected label behavior is specified.

## Relative Robustness

Report the clean score and perturbed score together:

$$
\mathrm{Drop}_{\mathcal{T}}
=
M_{\mathrm{clean}} - M_{\mathcal{T}}
$$

or as a ratio when the metric scale makes that clearer:

$$
\mathrm{Retention}_{\mathcal{T}}
=
\frac{M_{\mathcal{T}}}{M_{\mathrm{clean}}+\epsilon}
$$

For loss-like quantities, reverse the sign or report the gap explicitly. Do not mix "higher is better" and "lower is better" metrics without stating direction.

## Computational Biology Examples

| Area | Perturbation | Label-Preservation Caveat |
| --- | --- | --- |
| Molecule property | tautomer, protonation, salt, stereochemistry, conformer | chemical state may change the label |
| Docking | receptor preparation, side-chain state, box size, conformer seed | pose and score may legitimately change |
| Protein modeling | homolog removal, mutation, structure resolution, predicted structure input | mutation can alter function or stability |
| Virtual screening | decoy construction, library subset, failed docking treatment | active prevalence changes enrichment |
| Genome sequence | reverse complement, mutation, window shift | strand and context may change interpretation |

## Paper Claim Template

Use this shape when writing a paper note:

> The model is robust to `{perturbation}` at `{severity}` on `{slice}` under `{metric}`, with `{uncertainty}`. The perturbation is label-preserving because `{reason}`. Failure cases are `{failure_modes}`.

## Checks

- What perturbation is realistic for deployment?
- Does the perturbation preserve the correct label?
- Is robustness measured per subgroup, scaffold, family, source, or modality?
- Does the model fail gracefully or produce confident invalid outputs?
- Are robustness checks separated from training-time augmentation?
- Are failed preprocessing, failed inference, and invalid generations counted?
- Is the robustness comparison paired on the same examples?

## Related

- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/modalities/index|Modalities]]
- [[concepts/systems/inference|Inference]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
