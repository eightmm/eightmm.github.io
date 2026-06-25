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

## Key Ideas

- Active learning changes the data collection process, not just the model architecture.
- Uncertainty sampling can focus labeling on ambiguous points, but it may over-sample outliers.
- Diversity constraints prevent a batch from containing many near-duplicates.
- In molecular and protein work, acquisition must consider scaffold, family, assay, and distribution bias.

## Practical Checks

- What budget is available for the next labeling or evaluation round?
- Is the acquisition function aligned with the downstream metric?
- Are selected examples diverse enough to avoid redundant labels?
- Does the validation set remain fixed and unbiased while training data grows?

## Related

- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/learning/supervised-learning|Supervised learning]]
