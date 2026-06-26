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

## Checks

- Are models compared on the same examples?
- Is the metric stable under resampling?
- Does the resampling unit match the data dependence: example, molecule, target, assay, or cluster?
- Are confidence intervals reported for the metric difference, not only each metric?
- Is the test set large and diverse enough for bootstrap to be meaningful?

## Related

- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/effect-size|Effect size]]
- [[concepts/evaluation/metric|Metric]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
