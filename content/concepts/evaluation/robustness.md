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

## Checks

- What perturbation is realistic for deployment?
- Does the perturbation preserve the correct label?
- Is robustness measured per subgroup, scaffold, family, source, or modality?
- Does the model fail gracefully or produce confident invalid outputs?
- Are robustness checks separated from training-time augmentation?

## Related

- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/modalities/index|Modalities]]
- [[concepts/systems/inference|Inference]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
