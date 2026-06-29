---
title: Active Learning
tags:
  - active-learning
  - data
  - evaluation
---

# Active Learning

Active learning selects which unlabeled examples should be labeled next. It is useful when labels are expensive, such as assays, docking review, expert annotation, or manual evaluation of generated outputs.

At each round, a model scores unlabeled candidates $x\in\mathcal{U}$ with an acquisition function $a(x)$:

$$
x^\* = \arg\max_{x\in\mathcal{U}} a(x)
$$

Common acquisition signals include uncertainty, diversity, expected improvement, or disagreement between models.

For a batch of $k$ examples, the selection usually needs both informativeness and diversity:

$$
B^\*
=
\arg\max_{B\subset\mathcal{U},\ |B|=k}
\sum_{x\in B} a(x)
-
\lambda
\sum_{\substack{x_i,x_j\in B\\i<j}}
s(x_i,x_j)
$$

where $s(x_i,x_j)$ measures similarity and $\lambda$ controls redundancy penalty.

## Key Ideas

- Active learning changes the data collection process, not just the model architecture.
- Uncertainty sampling can focus labeling on ambiguous points, but it may over-sample outliers.
- Diversity constraints prevent a batch from containing many near-duplicates.
- In molecular and protein work, acquisition must consider scaffold, family, assay, and distribution bias.

## Acquisition Map

| Acquisition Signal | Useful When | Risk |
| --- | --- | --- |
| Uncertainty | labels are most useful near decision boundaries | outliers and poorly calibrated uncertainty dominate |
| Diversity | batch labels are expensive and redundancy is wasteful | diverse points may not be task-informative |
| Expected improvement | optimization over candidate utility matters | surrogate error can exploit the acquisition rule |
| Committee disagreement | multiple models expose epistemic uncertainty | ensemble diversity may be artificial |
| Coverage / representativeness | target distribution is broad | can ignore rare but important cases |

## Practical Checks

- What budget is available for the next labeling or evaluation round?
- Is the acquisition function aligned with the downstream metric?
- Are selected examples diverse enough to avoid redundant labels?
- Does the validation set remain fixed and unbiased while training data grows?
- Is the acquisition pool representative of the deployment or screening population?
- Are newly labeled examples tracked as a separate data-collection round?

## Related

- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/learning/supervised-learning|Supervised learning]]
