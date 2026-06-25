---
title: Bias-Variance Tradeoff
tags:
  - math
  - statistics
  - machine-learning
---

# Bias-Variance Tradeoff

The bias-variance tradeoff explains two different sources of prediction error: systematic underfitting and sensitivity to training data.

For a target $y=f^\*(x)+\epsilon$ and a learned predictor $\hat{f}(x)$, the expected squared error at $x$ can be decomposed as:

$$
\mathbb{E}\left[(\hat{f}(x)-y)^2\right]
=
\left(\mathbb{E}[\hat{f}(x)] - f^\*(x)\right)^2
+
\mathbb{E}\left[
(\hat{f}(x)-\mathbb{E}[\hat{f}(x)])^2
\right]
+
\sigma^2
$$

The three terms are squared bias, variance, and irreducible noise.

## Interpretation

- High bias: the model class or features are too limited.
- High variance: the model is too sensitive to the training sample.
- Noise: the target contains uncertainty the model cannot remove.

## In Modern ML

Deep learning does not always follow the classical simple U-shaped story, but the distinction still helps debug:

- underfitting vs overfitting
- data scarcity vs model capacity
- noisy labels vs optimization failure
- unstable validation metrics vs real model improvement

## Checks

- Is train error high? Suspect bias, optimization, or label mismatch.
- Is train error low but validation error high? Suspect variance, leakage, or distribution shift.
- Are metrics unstable across seeds or splits? Estimate variance before claiming improvement.
- Does more data help more than a larger model?

## Related

- [[concepts/math/statistical-estimator|Statistical estimator]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/evaluation/cross-validation|Cross-validation]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
