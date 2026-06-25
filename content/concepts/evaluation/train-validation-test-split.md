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
\mathcal{D}_{\mathrm{train}}
\cap
\mathcal{D}_{\mathrm{val}}
\cap
\mathcal{D}_{\mathrm{test}}
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

## Split Units

- Row split: useful only when rows are independent examples.
- Group split: required when examples share a molecule, protein, subject, scaffold, family, source, or time period.
- Temporal split: required when the claim is future performance.
- Domain split: required when the claim is transfer to a new source or deployment environment.

## Checks

- Is the test set used exactly once for final reporting?
- Are preprocessing, imputation, and normalization fit only on train data?
- Does the split group related molecules, proteins, structures, or time periods correctly?
- Does validation select the same kind of generalization expected at test time?
- Is the split unit documented alongside the metric?

## Related

- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
