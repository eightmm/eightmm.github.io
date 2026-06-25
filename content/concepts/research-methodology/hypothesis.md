---
title: Hypothesis
tags:
  - research
  - methodology
  - experiments
---

# Hypothesis

A hypothesis is a falsifiable claim about what should happen and why. It should be written before running the experiment.

A useful form is:

$$
H:
\quad
M(f_{\mathrm{new}}, S)
- M(f_{\mathrm{base}}, S)
\ge \delta
$$

where $M$ is a metric, $S$ is the evaluation split, and $\delta$ is the minimum effect size that would matter.

## Requirements

- It names the baseline.
- It names the metric and split.
- It states a threshold for success.
- It isolates one changed variable.
- It can fail.

## Anti-Patterns

- Writing the hypothesis after seeing results.
- Changing the metric or threshold after the run.
- Reporting only the best seed or checkpoint.
- Combining multiple changes and claiming causality.
- Treating a leaky split as evidence.

## Related

- [[concepts/research-methodology/research-question|Research question]]
- [[concepts/research-methodology/experiment-design|Experiment design]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
