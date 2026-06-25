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

## Checks

- Is the test set used exactly once for final reporting?
- Are preprocessing, imputation, and normalization fit only on train data?
- Does the split group related molecules, proteins, structures, or time periods correctly?
- Does validation select the same kind of generalization expected at test time?

## Related

- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
