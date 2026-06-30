---
title: Train Validation Test Split
tags:
  - evaluation
  - methodology
  - data
---

# Train Validation Test Split

A train/validation/test split separates model fitting, model selection, and final evaluation. Without this separation, the reported number can silently become a training signal.

The intended separation is:

$$
\mathcal{D}
= \mathcal{D}_{\mathrm{train}}
\cup \mathcal{D}_{\mathrm{val}}
\cup \mathcal{D}_{\mathrm{test}}
$$

$$
\mathcal{D}_{\mathrm{train}}\cap\mathcal{D}_{\mathrm{val}}
=
\mathcal{D}_{\mathrm{train}}\cap\mathcal{D}_{\mathrm{test}}
=
\mathcal{D}_{\mathrm{val}}\cap\mathcal{D}_{\mathrm{test}}
= \varnothing
$$

More precisely, the split should be disjoint at the unit that matters for generalization:

$$
u(x_i)=u(x_j)
\Rightarrow
s(x_i)=s(x_j)
$$

where $u$ maps an example to its split unit, such as molecule scaffold, protein family, patient, time period, source, or document group.

Training chooses parameters:

$$
\hat{\theta}
= \arg\min_\theta
\hat{R}_{\mathrm{train}}(\theta)
$$

Validation chooses hyperparameters or checkpoints:

$$
\hat{\lambda}
= \arg\min_\lambda
\hat{R}_{\mathrm{val}}(\hat{\theta}_{\lambda})
$$

Test evaluation estimates the final claim:

$$
\hat{R}_{\mathrm{test}}(\hat{\theta}_{\hat{\lambda}})
$$

## Split Contract

A split is part of the claim. The same metric means different things under different split units.

| Split field | Defines |
| --- | --- |
| example unit | one row, molecule, protein, structure, document, assay |
| split unit | what must not cross train/val/test |
| selection unit | what validation is allowed to tune |
| claim unit | what final performance is supposed to generalize to |

The contract should satisfy:

$$
\text{split unit}
\succeq
\text{claim unit}
$$

If the claim is performance on new scaffolds, the split cannot be random row-level. If the claim is performance on future data, the split should respect time.

## Split Units

- Row split: useful only when rows are independent examples.
- Group split: required when examples share a molecule, protein, subject, scaffold, family, source, or time period.
- Temporal split: required when the claim is future performance.
- Domain split: required when the claim is transfer to a new source or deployment environment.

## Common Split Failure Modes

| Failure | Consequence |
| --- | --- |
| duplicate entities cross splits | memorization appears as generalization |
| preprocessing fit before split | validation/test distribution leaks into train |
| hyperparameters tuned on test | final metric becomes model selection signal |
| random rows for grouped biology data | analogs, scaffolds, proteins, or assays leak |
| validation easier than test | selected model may not match final claim |
| temporal order ignored | future-performance claim is unsupported |

## What Belongs Where

| Use | Split |
| --- | --- |
| fit parameters | train |
| choose hyperparameters, checkpoint, threshold | validation |
| choose final claim after all decisions fixed | test |
| estimate uncertainty after final choice | bootstrap or repeated splits over test protocol |

Threshold selection is model selection. It belongs on validation unless the test protocol explicitly includes a nested threshold rule.

## Checks

- Is the test set used exactly once for final reporting?
- Are preprocessing, imputation, and normalization fit only on train data?
- Does the split group related molecules, proteins, structures, or time periods correctly?
- Does validation select the same kind of generalization expected at test time?
- Is the split unit documented alongside the metric?
- Are threshold selection, calibration, and early stopping all kept off the test set?
- Does the split support the exact wording of the generalization claim?

## Related

- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
