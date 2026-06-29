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

## Hypothesis Shape

Use a hypothesis to bind mechanism, intervention, evidence, and decision:

$$
H =
(\text{mechanism}, \text{change}, \text{baseline}, \text{metric}, \text{split}, \text{threshold})
$$

| Field | Question |
| --- | --- |
| Mechanism | Why should the change help? |
| Changed variable | What exactly differs from the baseline? |
| Baseline | What is the current reference system? |
| Metric | What observable will judge the claim? |
| Split | Where is the claim tested? |
| Threshold | How large must the effect be to matter? |
| Failure condition | What result would make the hypothesis weaker or false? |

## Prediction Before Running

Before running, write the expected direction and failure mode.

| Prediction | Example |
| --- | --- |
| positive | metric improves by at least $\delta$ on the target split |
| null | change does not move the metric beyond noise |
| negative | metric worsens, instability increases, or compute cost dominates |
| diagnostic | improvement appears only on an easy split or disappears under leakage checks |

The prediction matters because it prevents a result from being reinterpreted after the fact.

## Requirements

- It names the baseline.
- It names the metric and split.
- It states a threshold for success.
- It isolates one changed variable.
- It can fail.
- It states the expected failure mode.
- It separates primary metric from diagnostics.
- It names the dataset or benchmark version at the right level of precision.

## Claim Strength

The same hypothesis can support different claims depending on the evidence.

| Evidence | Supported Claim |
| --- | --- |
| small sanity run | implementation or mechanism is plausible |
| controlled ablation | one component likely matters under that protocol |
| repeated seeds | effect is less likely to be noise |
| stronger split | claim may generalize beyond near duplicates |
| external benchmark | claim may transfer to another data source |

Do not write a broad claim if the hypothesis only tests a narrow split, small subset, or diagnostic condition.

## Anti-Patterns

- Writing the hypothesis after seeing results.
- Changing the metric or threshold after the run.
- Reporting only the best seed or checkpoint.
- Combining multiple changes and claiming causality.
- Treating a leaky split as evidence.
- Confusing an engineering fix with a scientific mechanism.
- Treating a negative result as useless instead of narrowing the hypothesis.

## Related

- [[concepts/research-methodology/research-question|Research question]]
- [[concepts/research-methodology/experiment-design|Experiment design]]
- [[concepts/research-methodology/minimum-viable-experiment|Minimum viable experiment]]
- [[concepts/research-methodology/claim-evidence-record|Claim evidence record]]
- [[concepts/evaluation/effect-size|Effect size]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
