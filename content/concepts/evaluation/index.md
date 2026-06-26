---
title: Evaluation
tags:
  - evaluation
  - methodology
---

# Evaluation

Evaluation notes collect methods for measuring model quality honestly — how to split data, detect leakage, calibrate confidence, and test generalization beyond the training distribution.

The recurring theme: a metric is only as trustworthy as the split and protocol behind it. Most reported gains that fail to reproduce trace back to an evaluation flaw, not a modeling one.

The basic estimate is a held-out risk:

$$
\hat{R}(f)
= \frac{1}{m}\sum_{j=1}^{m}
\mathcal{L}(f(x_j), y_j)
$$

This number is only meaningful if the held-out set matches the intended generalization claim.

## Route Map

| Question | Start | Then Check |
| --- | --- | --- |
| what claim does the score support? | [Claim-evidence boundary](/concepts/evaluation/claim-evidence-boundary), [Benchmark claim contract](/concepts/evaluation/benchmark-claim-contract) | task, split, metric, baseline |
| is the evaluation set valid? | [Evaluation protocol](/concepts/evaluation/evaluation-protocol), [Evaluation set design](/concepts/evaluation/evaluation-set-design) | sampling, exclusions, contamination |
| which metric matches the task? | [Metric](/concepts/evaluation/metric), [Metric selection](/concepts/evaluation/metric-selection) | denominator, threshold, aggregation |
| classification or detection? | [Confusion matrix](/concepts/evaluation/confusion-matrix), [Precision and recall](/concepts/evaluation/precision-recall), [Classification metrics](/concepts/evaluation/classification-metrics) | prevalence and threshold |
| probabilistic prediction? | [Probability metrics](/concepts/evaluation/probability-metrics), [Proper scoring rule](/concepts/evaluation/proper-scoring-rule), [Brier score](/concepts/evaluation/brier-score), [Calibration](/concepts/evaluation/calibration) | reliability and NLL |
| ranking, retrieval, or screening? | [Ranking metrics](/concepts/evaluation/ranking-metrics), [Negative set](/concepts/evaluation/negative-set) | label completeness and candidate pool |
| regression or structure? | [Regression metrics](/concepts/evaluation/regression-metrics) | units, target scale, censoring, coordinate mapping |
| generation? | [Generation evaluation](/concepts/evaluation/generation-evaluation) | validity, diversity, novelty, task utility |
| is the improvement reliable? | [Baseline](/concepts/evaluation/baseline), [Paired comparison](/concepts/evaluation/paired-comparison), [Confidence interval](/concepts/evaluation/confidence-interval) | seed variance and effect size |
| could there be leakage? | [Leakage](/concepts/evaluation/leakage), [Test-set contamination](/concepts/evaluation/test-set-contamination), [Train/validation/test split](/concepts/evaluation/train-validation-test-split) | example unit and split unit |

## Method Groups

| Group | Notes |
| --- | --- |
| Benchmarks | [Benchmark](/concepts/data/benchmark), [Sampling strategy](/concepts/data/sampling-strategy), [Benchmark saturation](/concepts/evaluation/benchmark-saturation) |
| Statistical evidence | [Statistical estimator](/concepts/math/statistical-estimator), [Central limit theorem](/concepts/math/central-limit-theorem), [Hypothesis testing](/concepts/math/hypothesis-testing), [Bootstrap evaluation](/concepts/evaluation/bootstrap-evaluation) |
| Model comparison | [Baseline](/concepts/evaluation/baseline), [Ablation study](/concepts/evaluation/ablation-study), [Statistical significance](/concepts/evaluation/statistical-significance), [Multiple comparisons](/concepts/evaluation/multiple-comparisons) |
| Model selection | [Model selection](/concepts/machine-learning/model-selection), [Hyperparameter tuning](/concepts/machine-learning/hyperparameter-tuning), [Early stopping](/concepts/machine-learning/early-stopping), [Cross-validation](/concepts/evaluation/cross-validation) |
| Uncertainty | [Uncertainty estimation](/concepts/evaluation/uncertainty-estimation), [Reliability diagram](/concepts/evaluation/reliability-diagram), [Conformal prediction](/concepts/evaluation/conformal-prediction), [Selective prediction](/concepts/evaluation/selective-prediction) |
| Chem-bio risks | [Applicability domain](/concepts/evaluation/applicability-domain), [Assay harmonization](/concepts/evaluation/assay-harmonization), [Activity cliff](/concepts/evaluation/activity-cliff), [Scaffold split](/concepts/evaluation/scaffold-split), [Protein family split](/concepts/evaluation/protein-family-split), [Boltzmann ceiling analysis](/concepts/evaluation/boltzmann-ceiling) |
| Diagnosis | [Failure mode taxonomy](/concepts/evaluation/failure-mode-taxonomy), [Error analysis](/concepts/evaluation/error-analysis), [Robustness](/concepts/evaluation/robustness), [Interpretability](/concepts/evaluation/interpretability) |
| Reading papers | [Result table reading](/papers/analysis/result-table-reading), [Representation evaluation](/concepts/learning/representation-evaluation), [Linear probing](/concepts/learning/linear-probing), [Fine-tuning protocol](/concepts/learning/fine-tuning-protocol) |

## Related

- [[concepts/math/expectation|Expectation]]
- [[agents/verification/verification-loop|Verification loop]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
- [[concepts/index|Concepts]]
