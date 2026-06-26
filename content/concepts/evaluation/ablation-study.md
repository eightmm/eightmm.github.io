---
title: Ablation Study
tags:
  - evaluation
  - methodology
---

# Ablation Study

An ablation study removes or changes parts of a method to test which components actually matter. It is a way to move from "the full system works" to "this design choice explains the gain."

For a full model $f_{\mathrm{full}}$ and a model with component $c$ removed:

$$
\Delta_c
=
M(f_{\mathrm{full}})
- M(f_{\setminus c})
$$

where $M$ is the evaluation metric.

## Checks

- Is only one component changed at a time?
- Are all variants trained and evaluated under the same budget?
- Is variance across seeds or splits reported when the effect is small?
- Is the ablation interpreted with effect size, not only statistical significance?
- Were many ablations tried and only the strongest one emphasized?
- Does the ablation test the claimed mechanism, or only a weaker implementation?

## Related

- [[concepts/research-methodology/experiment-design|Experiment design]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/effect-size|Effect size]]
- [[concepts/evaluation/multiple-comparisons|Multiple comparisons]]
- [[papers/paper-review-workflow|Paper review workflow]]
- [[agents/verification/verification-loop|Verification loop]]
