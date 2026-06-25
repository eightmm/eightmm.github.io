---
title: Experiment Design
tags:
  - research
  - experiments
  - methodology
---

# Experiment Design

Experiment design defines the smallest run that could disprove a hypothesis. In ML research, a good experiment is controlled, cheap enough to run, and connected to a baseline.

One-variable comparison:

$$
\Delta M
= M(f_{\mathrm{changed}}) - M(f_{\mathrm{baseline}})
$$

The comparison is interpretable only when the changed variable is isolated.

## Design Elements

- Question.
- Hypothesis.
- Predicted outcome before running.
- Baseline.
- Single changed variable.
- Dataset version and split.
- Metric and success threshold.
- Budget and stop condition.

## Checks

- Could this run disprove the idea?
- Is a smaller probe possible before a full run?
- Are data, split, loss, metric, and seed policy fixed?
- Is the baseline strong enough?
- Are variance and repeated seeds needed?
- Will the result be recorded even if it is negative?

## Related

- [[concepts/research-methodology/hypothesis|Hypothesis]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/data/benchmark|Benchmark]]
- [[concepts/systems/training-run|Training run]]
