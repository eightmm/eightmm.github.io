---
title: Ablation Map
unlisted: true
aliases:
  - papers/ablation-map
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

## Map Table

Use a table instead of prose-only ablation notes.

| Component | Claimed role | Removal/replacement | Metric delta | Confounder | Verdict |
| --- | --- | --- | --- | --- | --- |
| architecture block | improves representation | remove block | $\Delta_i$ | parameter count changed | weak/medium/strong |
| loss term | stabilizes objective | set weight to zero | $\Delta_i$ | retuning missing | weak/medium/strong |
| data source | adds coverage | train without source | $\Delta_i$ | fewer samples | weak/medium/strong |
| preprocessing | removes noise | disable step | $\Delta_i$ | changes input distribution | weak/medium/strong |

The verdict should be about evidence for the component, not whether the paper is good overall.

## Paired View

When the same seeds or splits are available, compare paired results:

$$
d_k
=
M_k(A_{\mathrm{full}})
-
M_k(A_{\setminus c_i})
$$

Then inspect the mean and variance of $d_k$, not only a single headline delta.

## Checks

- Does the ablation isolate one component?
- Is the baseline strong enough without the component?
- Are multiple seeds or confidence intervals reported?
- Is the gain consistent across tasks or only one benchmark?
- Does the ablation support a mechanism claim or only an empirical difference?
- Are parameter count, training budget, data volume, and tuning budget controlled?
- Is the ablation interpreted within the tested benchmark scope?

## Related

- [[concepts/evaluation/ablation-study|Ablation study]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/analysis/limitation-taxonomy|Limitation taxonomy]]
- [[concepts/evaluation/effect-size|Effect size]]
- [[concepts/evaluation/seed-variance|Seed variance]]
- [[concepts/evaluation/multiple-comparisons|Multiple comparisons]]
- [[concepts/research-methodology/threat-to-validity|Threat to validity]]
