---
title: Paired Comparison
tags:
  - evaluation
  - statistics
  - methodology
---

# Paired Comparison

A paired comparison evaluates two systems on the same evaluation units and analyzes the per-unit difference. It is often stronger than comparing two independent aggregate scores because it removes variation from example difficulty.

For paired examples $i=1,\ldots,n$:

$$
d_i = m_i(f_A) - m_i(f_B)
$$

$m_i(f)$ is the per-example score for model $f$. The mean paired effect is:

$$
\bar{d} = \frac{1}{n}\sum_{i=1}^{n} d_i
$$

The uncertainty should be estimated over the correct paired unit: example, scaffold, protein family, target, document, user, prompt group, or benchmark case.

## When To Use

- Two models are evaluated on the same test examples.
- A paper claims model $A$ improves over baseline $B$.
- Per-example scores or predictions are available.
- The benchmark has heterogeneous difficulty.
- A bootstrap or significance test should estimate uncertainty in $\Delta M$ directly.

## Paired Bootstrap

Use the same resampled indices for both models:

$$
\Delta^{(b)}
=
M_A(\mathcal{D}^{(b)})
-
M_B(\mathcal{D}^{(b)})
$$

The distribution of $\Delta^{(b)}$ estimates the uncertainty of the improvement.

## Checks

- Are the two models evaluated on the exact same examples?
- Is the resampling unit independent enough for the claim?
- Are invalid outputs and missing predictions handled identically?
- Is the reported interval for the difference, not only separate intervals for each model?
- Are subgroup effects inspected when the mean improvement is small?

## Related

- [[concepts/evaluation/bootstrap-evaluation|Bootstrap evaluation]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/effect-size|Effect size]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[papers/analysis/evidence-table|Evidence table]]
