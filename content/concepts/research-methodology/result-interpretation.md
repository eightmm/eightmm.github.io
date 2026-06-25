---
title: Result Interpretation
tags:
  - research
  - methodology
  - evaluation
---

# Result Interpretation

Result interpretation compares what happened with what was predicted before the run. A result is not just a number; it is evidence for or against a hypothesis under a stated protocol.

The basic comparison is:

$$
\text{observed } \Delta M
\quad \text{vs.} \quad
\text{predicted } \Delta M_{\mathrm{expected}}
$$

## Outcomes

- Confirmed: the result meets the pre-registered success threshold.
- Not confirmed: the result does not meet the threshold.
- Surprising: the direction, size, or failure mode differs from prediction.
- Inconclusive: the run failed, variance is too large, or the protocol is flawed.

## Checks

- Was the metric computed on the intended split?
- Did any data, code, or environment drift occur?
- Is the effect larger than run-to-run variance?
- Does the result beat the baseline that matters?
- Is the conclusion limited to the tested setting?
- What should be updated: hypothesis, data, model, metric, or workflow?

## Related

- [[papers/claim-extraction|Claim extraction]]
- [[concepts/research-methodology/research-log|Research log]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/robustness|Robustness]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/evaluation/metric|Metric]]
