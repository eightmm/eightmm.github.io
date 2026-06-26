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

## Pairing Unit

The pair should be the smallest unit that is independent enough for the claim:

| Domain | Safer Paired Unit | Risk If Too Small |
| --- | --- | --- |
| classification/regression | test example or grouped example | duplicates inflate evidence |
| retrieval/ranking | query | candidate-level resampling breaks query structure |
| molecule tasks | scaffold group, assay source, or molecule depending on claim | analog series dominate |
| protein tasks | protein family, target, or complex group | homologs act like repeated evidence |
| generation | condition, prompt, target, or seed batch | cherry-picked samples dominate |
| agent/tool workflows | task instance or user request | repeated attempts on one task inflate success |

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

For grouped data, resample groups and keep all members inside a selected group:

$$
\mathcal{G}^{(b)}
\sim
\operatorname{Bootstrap}(\mathcal{G}),
\qquad
\mathcal{D}^{(b)}
=
\{z_i: g_i\in \mathcal{G}^{(b)}\}
$$

where $g_i$ is the group for example $z_i$. This is often more honest for scaffold, protein-family, target, prompt, or dataset-source comparisons.

## Interpretation

| Result | Interpretation |
| --- | --- |
| $\Delta$ large and interval excludes zero | stronger evidence of improvement under the tested population |
| $\Delta$ large but interval wide | promising but benchmark is too small/noisy |
| $\Delta$ small but statistically stable | real difference may be practically irrelevant |
| aggregate improves but subgroup regresses | claim must be narrowed or failure mode reported |
| only best seed/checkpoint compared | paired comparison is contaminated by model selection |

## Checks

- Are the two models evaluated on the exact same examples?
- Is the resampling unit independent enough for the claim?
- Are invalid outputs and missing predictions handled identically?
- Is the reported interval for the difference, not only separate intervals for each model?
- Are subgroup effects inspected when the mean improvement is small?
- Is the direction of the metric fixed so positive $\Delta$ always means better?
- Are ties, failed predictions, invalid generations, and abstentions included consistently?
- Was the final test set untouched until after selecting model, threshold, prompt, and checkpoint?

## Related

- [[concepts/evaluation/bootstrap-evaluation|Bootstrap evaluation]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/effect-size|Effect size]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/seed-variance|Seed variance]]
- [[concepts/evaluation/multiple-comparisons|Multiple comparisons]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[papers/analysis/evidence-table|Evidence table]]
