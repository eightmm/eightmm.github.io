---
title: Metric Selection
tags:
  - evaluation
  - metrics
  - methodology
---

# Metric Selection

Metric selection chooses which numbers should decide model quality for a task. It comes after [[concepts/tasks/task-specification|Task specification]] and before comparing models.

A metric should estimate the cost or utility that matters:

$$
M^\*
=
\arg\min_{M \in \mathcal{M}}
\left|
\mathrm{decision\ risk}(M)
-
\mathrm{target\ risk}
\right|
$$

where $\mathcal{M}$ is the candidate metric set. This is a conceptual rule: choose the metric whose behavior best matches the decision the result will support.

## Selection Order

1. Output space: class, scalar, ranking, sequence, mask, graph, coordinate, or action.
2. Decision: threshold, ranking, retrieval, generation, screening, regression, or human review.
3. Error cost: false positive, false negative, invalid output, low utility, miscalibration, or unsafe action.
4. Split: IID, temporal, source, scaffold, protein-family, or deployment-like split.
5. Primary metric: the number used for model selection or final comparison.
6. Diagnostics: calibration, uncertainty, subgroup error, robustness, or error analysis.

## Common Choices

| Task behavior | Primary metric candidates | Diagnostics |
| --- | --- | --- |
| Hard classification | accuracy, balanced accuracy, F1 | confusion matrix, threshold sweep |
| Imbalanced detection | AUPRC, recall at precision, F1 | calibration, false-positive analysis |
| Probability quality | NLL, Brier score, ECE | reliability diagram |
| Regression | MAE, RMSE, $R^2$ | residual analysis, calibration if probabilistic |
| Ranking/retrieval | MRR, NDCG, recall@k, precision@k | query-slice error analysis |
| Generation | validity, utility, diversity, novelty | human or task-specific checks |
| Structure/pose | RMSD, clash/geometry checks, pose plausibility | per-complex failure analysis |
| Agent/tool use | task success, tool-call validity, cost, recovery | trace review and safety checks |

## Primary vs Diagnostic

The primary metric should answer the main comparison question. Diagnostics explain why the primary metric moved.

For a model family $\mathcal{F}$ and validation metric $M_{\mathrm{primary}}$:

$$
f^\*
=
\arg\max_{f \in \mathcal{F}}
M_{\mathrm{primary}}(f; \mathcal{D}_{\mathrm{val}})
$$

Diagnostics should not silently become selection criteria unless the protocol says so before evaluation.

## Checks

- What decision will be made from the metric?
- Is the output space compatible with the metric?
- Are invalid outputs counted, filtered, repaired, or ignored?
- Is the metric robust to class imbalance, duplicates, or grouped examples?
- Is calibration needed because downstream use depends on confidence?
- Is the primary metric named before model selection?
- Are diagnostic metrics separated from selection metrics?
- Does the metric remain valid under the chosen split and deployment boundary?

## Related

- [[concepts/evaluation/metric|Metric]]
- [[concepts/machine-learning/decision-rule|Decision rule]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/failure-mode-taxonomy|Failure mode taxonomy]]
- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/tasks/task-output-space|Task output space]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
- [[concepts/evaluation/error-analysis|Error analysis]]
