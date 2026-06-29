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

## Decision First

An MVE should answer one decision, not produce a final paper result.

| Decision | Minimum Evidence |
| --- | --- |
| abandon idea | sanity check fails, baseline is already enough, or cost is unjustified |
| debug implementation | tiny subset confirms shapes, loss, gradients, and outputs |
| scale experiment | small run shows signal and stable failure modes |
| narrow claim | result works only under a specific split, data source, or metric |
| write public note | result can be generalized without private data or unpublished details |

If the result cannot change a decision, it is not a minimum viable experiment.

## MVE Contract

Write the contract before launching:

$$
E_{\min}
=
(\text{question}, \text{hypothesis}, \text{baseline}, \text{probe}, \text{metric}, \text{decision rule})
$$

| Field | Include |
| --- | --- |
| Question | one uncertainty to reduce |
| Hypothesis | falsifiable expected result |
| Baseline | cheapest credible comparison |
| Probe | smallest dataset/run/subset that can reveal the signal |
| Metric | one primary metric and a few diagnostics |
| Decision rule | pass, fail, inconclusive, and next action |
| Public boundary | what can be written without exposing private material |

## Size Ladder

Scale only after the cheaper rung changes the decision.

| Rung | Purpose |
| --- | --- |
| static inspection | check data schema, labels, formulas, and leakage risks |
| tiny smoke run | check implementation and artifact flow |
| small public subset | check whether signal exists under a public-safe setting |
| controlled ablation | isolate one changed component |
| repeated-seed run | estimate variance around the effect |
| full benchmark | support a benchmark-level claim |

The full benchmark should not be the first experiment unless cheaper probes cannot answer the question.

## Design Rules

- Test one uncertainty at a time.
- Use a cheap baseline before a complex model.
- Prefer a small public subset before a full-scale run.
- Define pass, fail, and inconclusive outcomes before running.
- Record the result even when it is negative.
- Keep the run artifact small enough to inspect.
- Use public data or sanitized summaries if the result may become a blog/wiki note.

## Checks

- What decision will this experiment change?
- What observation would falsify the hypothesis?
- Is the experiment smaller than the final intended run?
- Is there a baseline or sanity check?
- Is the result publishable as a general lesson without private details?
- Are pass, fail, and inconclusive outcomes defined before launch?
- Is the experiment testing one uncertainty or many?
- Does the run record include enough context to explain the result later?

## Result Handling

| Outcome | Action |
| --- | --- |
| pass | scale one rung or convert the lesson into a claim-evidence record |
| fail | record a negative result and revise the hypothesis |
| inconclusive | identify missing evidence instead of choosing the best-looking interpretation |
| broken run | separate infrastructure/code failure from hypothesis evidence |

## Related

- [[concepts/research-methodology/hypothesis|Hypothesis]]
- [[concepts/research-methodology/experiment-design|Experiment design]]
- [[concepts/research-methodology/negative-result|Negative result]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
- [[concepts/research-methodology/claim-evidence-record|Claim evidence record]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[papers/reproducibility/reproduction-plan|Reproduction plan]]
- [[concepts/evaluation/baseline|Baseline]]
