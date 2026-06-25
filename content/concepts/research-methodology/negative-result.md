---
title: Negative Result
tags:
  - research
  - experiments
  - methodology
---

# Negative Result

A negative result is an experiment that does not support the hypothesis, does not beat the baseline, or reveals a failure mode. It is valuable when the setup and interpretation are clear.

For a planned improvement threshold $\tau$, a negative result may be:

$$
\Delta M
=
M_{\mathrm{new}} - M_{\mathrm{baseline}}
< \tau
$$

where $M$ is the target metric.

## Types

- Null result: no meaningful change.
- Regression: metric gets worse.
- Instability: result depends strongly on seed, split, or configuration.
- Failure mode: the method works on average but fails on a critical subset.
- Invalid run: the experiment cannot answer the question because the protocol was flawed.

## Why It Matters

- Prevents repeating failed ideas.
- Clarifies assumptions and boundary conditions.
- Improves future experiment design.
- Makes public writing more credible than only reporting wins.

## Public-Safe Reporting

A public negative-result note should report the general lesson and methodology without exposing private datasets, internal targets, unpublished metrics, collaborator details, or private infrastructure.

## Checks

- Was the run capable of testing the hypothesis?
- Is the baseline strong and correctly computed?
- Is the result negative, inconclusive, or invalid?
- Is the failure tied to data, model, optimization, evaluation, or systems?
- What decision changes because of the result?

## Related

- [[concepts/research-methodology/result-interpretation|Result interpretation]]
- [[concepts/research-methodology/experiment-ledger|Experiment ledger]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[concepts/evaluation/baseline|Baseline]]
- [[logs/public-log-format|Public log format]]
