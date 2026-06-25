---
title: Ablation Map
tags:
  - papers
  - ablation
  - evaluation
---

# Ablation Map

An ablation map links each component of a method to the experiment that tests whether the component matters.

For a method with components $C=\{c_1,\ldots,c_k\}$, an ablation compares:

$$
\Delta_i
= M(A_{\mathrm{full}}) - M(A_{\setminus c_i})
$$

$M$ is the metric, $A_{\mathrm{full}}$ is the full method, and $A_{\setminus c_i}$ is the method with component $c_i$ removed or replaced.

## What To Map

- Component: architecture block, loss term, data source, feature, preprocessing step, or training trick.
- Claim: what the paper says the component contributes.
- Ablation: the experiment that isolates the component.
- Metric: the number affected by the ablation.
- Scope: whether the result holds across datasets, seeds, tasks, or model sizes.
- Confounder: what else changed besides the component.

## Checks

- Does the ablation isolate one component?
- Is the baseline strong enough without the component?
- Are multiple seeds or confidence intervals reported?
- Is the gain consistent across tasks or only one benchmark?
- Does the ablation support a mechanism claim or only an empirical difference?

## Related

- [[concepts/evaluation/ablation-study|Ablation study]]
- [[papers/evidence-table|Evidence table]]
- [[papers/claim-extraction|Claim extraction]]
- [[papers/limitation-taxonomy|Limitation taxonomy]]
- [[concepts/research-methodology/threat-to-validity|Threat to validity]]
