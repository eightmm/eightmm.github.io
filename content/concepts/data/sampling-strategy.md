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

## Dataset Sampling vs Batch Sampling

Two distributions are often involved:

$$
q_{\mathrm{dataset}}(x,y)
\ne
q_{\mathrm{batch}}(x,y)
$$

The dataset sampling distribution defines what examples exist. The batch sampling distribution defines what the optimizer sees at each step. Balanced batches can help optimization but should not make evaluation look like the balanced training distribution unless that is the target population.

## Importance Weighting

If training samples from $q$ while the desired risk is under $p$, a weighted objective may be needed:

$$
R_p(f)
=
\mathbb{E}_{(x,y)\sim q}
\left[
\frac{p(x,y)}{q(x,y)}
\mathcal{L}(f(x),y)
\right]
$$

In practice, $p/q$ may be unknown, so reports should state whether metrics are measured under natural, balanced, source-weighted, or task-weighted distributions.

## Failure Modes

- Oversampling rare positives improves training but evaluation silently uses a different prevalence.
- Hard negatives are actually unlabeled positives.
- Temporal sampling leaks future distribution information.
- Batch construction changes class, source, scaffold, family, or user proportions.
- Evaluation samples only easy cases and hides deployment tails.

## Checks

- What population does the sample represent?
- Are rare classes oversampled during training but not evaluation?
- Is the sample biased relative to the target population?
- Are hard negatives true negatives or unlabeled positives?
- Does sampling hide time, source, assay, scaffold, or family bias?
- Are batch construction rules part of the reproducible method?
- Is evaluation measured under the natural or rebalanced distribution?
- Are sample weights, class weights, and batch sampler rules reported separately?
- Are hard negatives verified as true negatives or labeled as negative-source-specific?

## Related

- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/data/class-imbalance|Class imbalance]]
- [[concepts/data/sampling-bias|Sampling bias]]
- [[concepts/data/data-distribution|Data distribution]]
- [[concepts/machine-learning/stochastic-gradient|Stochastic gradient]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
