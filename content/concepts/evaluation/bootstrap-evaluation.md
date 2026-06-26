---
title: Bootstrap Evaluation
tags:
  - evaluation
  - statistics
  - uncertainty
---

# Bootstrap Evaluation

Bootstrap evaluation estimates uncertainty in a metric by repeatedly resampling the evaluation set with replacement.

Given a test set $\mathcal{D}_{\mathrm{test}}$ and metric $M$, sample bootstrap datasets:

$$
\mathcal{D}_{\mathrm{test}}^{(b)}
\sim
\operatorname{Bootstrap}(\mathcal{D}_{\mathrm{test}})
$$

Then compute:

$$
\hat{M}^{(b)}
=
M(\mathcal{D}_{\mathrm{test}}^{(b)})
$$

The empirical distribution of $\{\hat{M}^{(b)}\}_{b=1}^{B}$ gives confidence intervals or uncertainty estimates.

## Percentile Interval

A common interval uses bootstrap quantiles:

$$
\left[
Q_{\alpha/2}(\{\hat{M}^{(b)}\}),
Q_{1-\alpha/2}(\{\hat{M}^{(b)}\})
\right]
$$

where $Q_q$ is the empirical $q$-quantile. This interval describes variability under the resampling design, not every possible deployment condition.

## Paired Bootstrap

For comparing two models $A$ and $B$ on the same examples, resample examples once and compute the paired difference:

$$
\Delta^{(b)}
=
M_A(\mathcal{D}^{(b)})
-
M_B(\mathcal{D}^{(b)})
$$

This estimates uncertainty in the improvement directly.

## Resampling Unit

| Data Structure | Resample |
| --- | --- |
| IID examples | examples |
| classification with duplicate entities | entity groups or deduplicated examples |
| molecular property prediction | scaffold groups, molecule groups, or assay-source groups |
| protein modeling | protein families, targets, or complexes |
| retrieval | queries, not candidate rows |
| generation | prompts/conditions and all samples generated for each condition |
| repeated seeds | seed/run units when estimating run variation |

If the resampling unit is too small, the interval looks narrower than the evidence supports.

## Failure Modes

| Failure | Consequence |
| --- | --- |
| bootstrap after test-set selection | interval ignores model-selection bias |
| resample candidates instead of queries | retrieval metrics become overconfident |
| resample ligand rows despite scaffold dependence | molecular benchmark appears stronger than it is |
| ignore failed/invalid outputs | generation methods look cleaner than attempted-sample reality |
| use bootstrap for tiny rare subgroup | interval may be unstable; report numerator/denominator too |

## Checks

- Are models compared on the same examples?
- Is the metric stable under resampling?
- Does the resampling unit match the data dependence: example, molecule, target, assay, or cluster?
- Are confidence intervals reported for the metric difference, not only each metric?
- Is the test set large and diverse enough for bootstrap to be meaningful?
- Is the number of bootstrap replicates and random seed recorded?
- Are all preprocessing, filtering, invalid outputs, and missing predictions inside the metric computation?
- Is the bootstrap used for final evidence, not for repeatedly choosing the best model?

## Related

- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/effect-size|Effect size]]
- [[concepts/evaluation/seed-variance|Seed variance]]
- [[concepts/evaluation/metric|Metric]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
