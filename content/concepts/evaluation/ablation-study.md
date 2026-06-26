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

## Claim Structure

An ablation should connect a design claim to evidence:

$$
\text{component } c
\rightarrow
\text{mechanism}
\rightarrow
\Delta M_c
$$

If the claim is "component $c$ improves long-context reasoning," the ablation should test long-context behavior, not only a broad aggregate score.

## One-Factor vs Interaction

A one-factor ablation changes one component:

$$
f_{\setminus c}
$$

But systems often have interactions. For components $a$ and $b$:

$$
\Delta_{a,b}
=
M(f_{\mathrm{full}})
-M(f_{\setminus a,\setminus b})
$$

The interaction effect is:

$$
I_{a,b}
=
\Delta_{a,b}
-\Delta_a
-\Delta_b
$$

If $I_{a,b}$ is large, components should not be interpreted independently.

## Budget Control

All variants should use comparable training and tuning budget:

$$
B_{\mathrm{full}}
\approx
B_{\setminus c}
$$

where $B$ includes data, optimizer steps, search trials, model selection rules, compute, and preprocessing. Otherwise the ablation may measure budget rather than component value.

## Negative Ablations

A useful ablation can show no effect:

$$
|\Delta_c| \approx 0
$$

This is evidence that the component may be unnecessary, redundant, or only useful in a different regime. Reporting only positive ablations overstates understanding.

## Checks

- Is only one component changed at a time?
- Are all variants trained and evaluated under the same budget?
- Is variance across seeds or splits reported when the effect is small?
- Is the ablation interpreted with effect size, not only statistical significance?
- Were many ablations tried and only the strongest one emphasized?
- Does the ablation test the claimed mechanism, or only a weaker implementation?
- Are interactions between components checked when the system is modular?
- Is the ablation result reported even when it weakens the story?
- Are invalid or failed variants counted consistently?
- Does the ablation use the same primary metric as the main claim?

## Related

- [[concepts/research-methodology/experiment-design|Experiment design]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/effect-size|Effect size]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/multiple-comparisons|Multiple comparisons]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[papers/paper-review-workflow|Paper review workflow]]
- [[papers/ablation-map|Ablation map]]
- [[agents/verification/verification-loop|Verification loop]]
