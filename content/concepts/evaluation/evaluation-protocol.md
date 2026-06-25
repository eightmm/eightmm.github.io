---
title: Evaluation Protocol
tags:
  - evaluation
  - methodology
  - protocol
---

# Evaluation Protocol

An evaluation protocol is the full procedure that turns data, models, metrics, and splits into a defensible claim. A metric alone is not a protocol.

A protocol can be summarized as:

$$
\mathcal{P}
=
(\mathcal{D}, s, f_\theta, \mathcal{M}, \mathcal{C})
$$

where $\mathcal{D}$ is the dataset, $s$ is the split rule, $f_\theta$ is the trained model, $\mathcal{M}$ is the metric set, and $\mathcal{C}$ is the claim being tested.

## Required Parts

- Data definition: example unit, label semantics, source, filtering, and preprocessing.
- Split rule: random, temporal, scaffold, protein-family, source-based, or other grouped split.
- Training rule: objective, model selection, early stopping, and hyperparameter search space.
- Metric rule: primary metric, secondary diagnostics, confidence intervals, and aggregation.
- Claim boundary: what population, task, and deployment setting the result supports.

## Model Selection vs Final Test

Validation is allowed to influence model choice:

$$
\lambda^\*
=
\arg\min_{\lambda}
\hat{R}_{\mathrm{val}}(f_{\theta(\lambda)})
$$

The final test estimate should evaluate the fixed choice:

$$
\hat{R}_{\mathrm{test}}(f_{\theta(\lambda^\*)})
$$

If the test set is repeatedly used for choices, it becomes part of the training loop.

## Evidence Strength

The protocol should match the claim:

- Random split supports interpolation over similar examples.
- Scaffold split supports chemotype shift claims.
- Protein-family split supports homology-shift claims.
- Temporal split supports future-data claims.
- Source split supports cross-lab or cross-dataset claims.

## Checks

- Is the example unit clear?
- Is the split unit aligned with the generalization claim?
- Are preprocessing steps fit only on train data?
- Is the primary metric named before looking at results?
- Are uncertainty, calibration, and error analysis included when decisions depend on confidence?
- Are limitations stated as part of the result?

## Related

- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/label-semantics|Label semantics]]
