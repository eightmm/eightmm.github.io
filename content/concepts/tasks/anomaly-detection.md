---
title: Anomaly Detection
tags:
  - tasks
  - anomaly-detection
  - evaluation
---

# Anomaly Detection

Anomaly detection identifies examples that deviate from expected behavior. It is used in monitoring, fraud, data quality, experimental screening, system logs, and model-output triage.

An anomaly detector often assigns a score:

$$
s(x)=\operatorname{anomaly\_score}(x)
$$

Predictions are made by thresholding:

$$
\hat{y}
=
\mathbf{1}[s(x) > \tau]
$$

where $\tau$ controls the tradeoff between false positives and false negatives.

If review capacity is limited, the operational objective is often top-k triage rather than binary classification:

$$
\operatorname{TopK}(\mathcal{D})
=
\operatorname{argsort}_{x\in\mathcal{D}} s(x)[1:k]
$$

In that setting, precision at $k$, recall at $k$, alert load, and time-to-detection may matter more than a global threshold.

## Key Ideas

- Anomalies can be rare, ambiguous, evolving, or defined only by operational cost.
- Supervised anomaly detection needs labeled anomalies; unsupervised methods estimate unusualness from normal data.
- A high anomaly score is not automatically a causal explanation.
- Evaluation is sensitive to class imbalance and incomplete labels.
- In monitoring workflows, alert fatigue can matter more than raw AUROC.

## Task Variants

| Variant | Output | Typical Metric Risk |
| --- | --- | --- |
| binary detection | anomaly / normal | threshold and prevalence dominate |
| ranking / triage | sorted candidate list | label incompleteness hides false negatives |
| time-series alerting | event time or alert window | delayed labels and alert fatigue |
| data-quality check | suspicious record or batch | root cause may be outside the example |
| model-output triage | generated output or prediction flagged for review | score is not an explanation |

## Practical Checks

- Are anomaly labels complete, partial, delayed, or proxy labels?
- Is the threshold chosen on validation data and frozen for test evaluation?
- Is the metric aligned with review budget or intervention cost?
- Does the anomaly definition drift over time?
- Are anomalies grouped by event, user, molecule, run, or target rather than independent examples?
- Is the base rate reported for each split or deployment slice?
- Are repeated alerts for the same underlying event deduplicated before evaluation?

## Related

- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/precision-recall|Precision and recall]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
