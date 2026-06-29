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

## Evaluation View

Bias and variance apply to metrics as well as predictors. A benchmark estimate can have low variance but high bias if the evaluation set is systematically easier than deployment, or low bias but high variance if it is small.

| Symptom | Possible interpretation |
| --- | --- |
| all models score poorly | high bias, wrong features, wrong task framing, or label mismatch |
| train score high, test score low | high variance, leakage-sensitive shortcut, or distribution shift |
| seed-to-seed ranking changes | metric variance may exceed model difference |
| larger model helps train but not test | capacity increases variance or exposes data limits |
| more data improves both train and test | previous regime was data-limited |

For metric estimates:

$$
\hat{M}
=
M + \operatorname{bias}(\hat{M}) + \eta,
\qquad
\operatorname{Var}(\eta)>0
$$

Small benchmark gains should be compared with variance from seeds, folds, paired bootstrap, or repeated runs before turning them into claims.

## Practical Levers

| Problem | Typical levers |
| --- | --- |
| high bias | better representation, larger model class, better objective, label semantics audit |
| high variance | more data, regularization, simpler model, stronger split controls, ensembling |
| high label noise | replicate handling, uncertainty, robust loss, clean subset analysis |
| unstable metric | larger evaluation set, paired comparison, confidence interval |

## Checks

- Is train error high? Suspect bias, optimization, or label mismatch.
- Is train error low but validation error high? Suspect variance, leakage, or distribution shift.
- Are metrics unstable across seeds or splits? Estimate variance before claiming improvement.
- Does more data help more than a larger model?
- Is the reported gain larger than metric and seed variance?
- Is the test set representative, or is estimator bias the dominant problem?

## Related

- [[concepts/math/statistical-estimator|Statistical estimator]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/evaluation/cross-validation|Cross-validation]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
