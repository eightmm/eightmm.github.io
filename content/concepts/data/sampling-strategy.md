---
title: Sampling Strategy
tags:
  - data
  - sampling
  - evaluation
---

# Sampling Strategy

Sampling strategy defines which examples enter the dataset and how batches are drawn during training or evaluation. It controls [[concepts/data/class-imbalance|class balance]], domain coverage, temporal bias, and rare-case visibility.

The training objective is usually an expectation under the sampling distribution:

$$
\mathbb{E}_{(x,y)\sim q_{\mathrm{train}}}
\left[\mathcal{L}(f_\theta(x),y)\right]
$$

If $q_{\mathrm{train}}$ differs from the deployment distribution $p_{\mathrm{deploy}}$, evaluation must name and test that shift.

## Common Strategies

- IID random sampling from a fixed dataset.
- Stratified sampling to preserve class or group proportions.
- Balanced sampling to expose rare labels.
- Hard negative sampling for retrieval and contrastive learning.
- Temporal sampling for data collected over time.
- Group-aware sampling to avoid related examples crossing splits.

## Checks

- What population does the sample represent?
- Are rare classes oversampled during training but not evaluation?
- Is the sample biased relative to the target population?
- Are hard negatives true negatives or unlabeled positives?
- Does sampling hide time, source, assay, scaffold, or family bias?
- Are batch construction rules part of the reproducible method?

## Related

- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/data/class-imbalance|Class imbalance]]
- [[concepts/data/sampling-bias|Sampling bias]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
