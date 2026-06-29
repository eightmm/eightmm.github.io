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

## Design Contract

An experiment should be inspectable as a contract:

$$
E =
(\mathcal{D}, f_{\mathrm{base}}, f_{\mathrm{changed}}, \mathcal{L}, M, S, B, R)
$$

where $\mathcal{D}$ is the dataset version, $f_{\mathrm{base}}$ is the baseline, $f_{\mathrm{changed}}$ is the changed system, $\mathcal{L}$ is the training objective, $M$ is the evaluation metric, $S$ is the split, $B$ is the budget, and $R$ is the run record.

## Control Table

| Control | Why It Matters |
| --- | --- |
| dataset version | avoids comparing across hidden data changes |
| split and seed policy | separates signal from split or seed luck |
| preprocessing | prevents train/test leakage and representation drift |
| baseline implementation | weak baselines exaggerate claims |
| changed variable | makes the causal interpretation possible |
| metric and threshold | prevents post-hoc metric selection |
| compute budget | prevents unfair search or tuning advantage |
| stopping rule | prevents peeking or selective continuation |

## Design Elements

- Question.
- Hypothesis.
- Predicted outcome before running.
- Baseline.
- Single changed variable.
- Dataset version and split.
- Metric and success threshold.
- Budget and stop condition.

## Pass, Fail, Inconclusive

Define the interpretation before running.

| Outcome | Condition | Next Step |
| --- | --- | --- |
| pass | primary metric clears threshold and diagnostics do not contradict it | scale, repeat, or write a bounded claim |
| fail | primary metric misses threshold or failure mode contradicts mechanism | revise or abandon hypothesis |
| inconclusive | variance, bug, data issue, or missing baseline blocks interpretation | run a narrower diagnostic |
| invalid | leakage, wrong split, broken artifact, or changed protocol | do not use as hypothesis evidence |

## Evidence Boundary

An experiment can support only what its protocol tests:

$$
\text{claim}
\subseteq
(\text{dataset}, \text{split}, \text{metric}, \text{baseline}, \text{budget})
$$

If the experiment uses a small public subset, the supported claim should mention that subset. If it uses a private run, the public note should preserve the method lesson but not publish unpublished results.

## Checks

- Could this run disprove the idea?
- Is a smaller probe possible before a full run?
- Are data, split, loss, metric, and seed policy fixed?
- Is the baseline strong enough?
- Are variance and repeated seeds needed?
- Will the result be recorded even if it is negative?
- Does the design distinguish implementation failure from hypothesis failure?
- Does the protocol state what evidence would be enough to stop?
- Is the public claim narrower than or equal to the experiment evidence?

## Related

- [[concepts/research-methodology/hypothesis|Hypothesis]]
- [[concepts/research-methodology/minimum-viable-experiment|Minimum viable experiment]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
- [[concepts/research-methodology/threat-to-validity|Threat to validity]]
- [[concepts/research-methodology/claim-evidence-record|Claim evidence record]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/data/benchmark|Benchmark]]
- [[concepts/systems/training-run|Training run]]
