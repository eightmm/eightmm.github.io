---
title: Threat to Validity
tags:
  - research
  - methodology
  - evaluation
---

# Threat To Validity

A threat to validity is a reason a result may not support the conclusion being drawn from it.

A result supports a claim only under assumptions:

$$
\text{evidence} + \text{assumptions}
\Rightarrow
\text{claim}
$$

A threat weakens the implication by attacking the data, measurement, comparison, implementation, or interpretation.

## Common Threats

- Data threat: biased sampling, missing labels, duplicated examples, or untracked preprocessing.
- Split threat: train/test overlap, random split where entity-level split is needed, or temporal leakage.
- Metric threat: metric does not match the actual task.
- Baseline threat: comparison is too weak or not tuned fairly.
- Implementation threat: result depends on a hidden bug, environment difference, or unreproducible setting.
- External validity threat: the result holds on a benchmark but not on intended deployment data.
- Interpretation threat: a mechanistic explanation goes beyond the evidence.

## Checks

- Which assumption would break the conclusion?
- Is the threat severe enough to narrow the claim?
- Can a small experiment test the threat?
- Does the paper or experiment report enough evidence to rule it out?
- Should the limitation be added to [[papers/limitation-taxonomy|Limitation taxonomy]]?

## Related

- [[papers/limitation-taxonomy|Limitation taxonomy]]
- [[papers/ablation-map|Ablation map]]
- [[papers/evidence-table|Evidence table]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
