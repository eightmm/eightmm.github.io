---
title: Failure Mode Taxonomy
tags:
  - evaluation
  - diagnostics
  - methodology
---

# Failure Mode Taxonomy

A failure mode taxonomy names how a model fails before trying to fix it. Aggregate metrics hide whether a system is wrong, invalid, overconfident, brittle, leaking, or out of scope.

For an example $(x,y)$, a failure label can be written as:

$$
z_i
=
g(x_i, y_i, \hat{y}_i, c_i, e_i)
$$

where $\hat{y}_i$ is the model output, $c_i$ is confidence or uncertainty, $e_i$ is environment or metadata context, and $g$ maps the case to a failure category.

## Common Failure Modes

- Wrong output: prediction is valid but incorrect.
- Invalid output: syntax, schema, geometry, action, or constraint violation.
- Low-utility output: technically valid but not useful for the downstream decision.
- Miscalibrated confidence: confidence does not match empirical correctness.
- Unsupported input: input lies outside the model's applicability domain.
- Robustness failure: small realistic perturbations change the answer too much.
- Retrieval failure: needed evidence or candidate is missing from retrieved context.
- Alignment failure: output refers to the wrong region, frame, entity, residue, ligand, or tool result.
- Leakage failure: evaluation used information unavailable at deployment.
- System failure: timeout, memory issue, tool error, missing artifact, or broken preprocessing.

## Failure Table

| Failure mode | Typical evidence | Follow-up |
| --- | --- | --- |
| Wrong output | metric error, false positive, false negative | error analysis, data review, model change |
| Invalid output | failed validity check | constrained decoding, validation, repair policy |
| Miscalibration | high confidence wrong cases | calibration, uncertainty estimation |
| OOD failure | split or subgroup gap | applicability domain, split redesign |
| Robustness failure | perturbation gap | augmentation, robustness evaluation |
| Leakage failure | deployment-unavailable information | protocol redesign |
| System failure | run or serving error | observability, retry, fallback |

## Counting Failures

Failure rates can be measured per category:

$$
\hat{p}_k
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf{1}[z_i = k]
$$

where $z_i$ is the assigned failure category and $k$ is one failure mode. Multi-label failures are allowed when one example has both invalid output and high confidence.

## Checks

- Are invalid outputs counted as failures rather than hidden by filtering?
- Are false positives and false negatives separated?
- Are confidence errors separated from correctness errors?
- Are failures sliced by modality, source, split group, scaffold, protein family, or task type?
- Is the failure caused by data, representation, model, objective, decoding, metric, or system behavior?
- Does each failure category suggest a different next experiment?

## Related

- [[concepts/evaluation/error-analysis|Error analysis]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/robustness|Robustness]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[agents/verification/verification-loop|Verification loop]]
