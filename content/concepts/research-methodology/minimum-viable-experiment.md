---
title: Minimum Viable Experiment
tags:
  - research
  - methodology
  - experiments
---

# Minimum Viable Experiment

A minimum viable experiment is the smallest experiment that can change the decision about a hypothesis.

For a hypothesis $H$, choose an experiment $E$ that maximizes information per cost:

$$
E^\star
= \arg\max_{E \in \mathcal{E}}
\frac{I(H; Y_E)}{\operatorname{cost}(E)}
$$

$Y_E$ is the result of experiment $E$, and $I(H;Y_E)$ is the information the result gives about the hypothesis.

## Design Rules

- Test one uncertainty at a time.
- Use a cheap baseline before a complex model.
- Prefer a small public subset before a full-scale run.
- Define pass, fail, and inconclusive outcomes before running.
- Record the result even when it is negative.

## Checks

- What decision will this experiment change?
- What observation would falsify the hypothesis?
- Is the experiment smaller than the final intended run?
- Is there a baseline or sanity check?
- Is the result publishable as a general lesson without private details?

## Related

- [[concepts/research-methodology/hypothesis|Hypothesis]]
- [[concepts/research-methodology/experiment-design|Experiment design]]
- [[concepts/research-methodology/negative-result|Negative result]]
- [[papers/reproducibility/reproduction-plan|Reproduction plan]]
- [[concepts/evaluation/baseline|Baseline]]
