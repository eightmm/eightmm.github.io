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

## Checks

- What distance is used: fingerprint Tanimoto, sequence identity, embedding distance, or structural similarity?
- Is the threshold chosen on validation data, not test data?
- Are metrics reported separately for in-domain and out-of-domain examples?
- Does confidence degrade under distribution shift?
- Is calibration checked inside and outside the applicability domain?

## Related

- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/evaluation/metric|Metric]]
