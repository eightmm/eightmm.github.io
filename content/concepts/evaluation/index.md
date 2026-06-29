---
title: Evaluation
tags:
  - evaluation
  - methodology
---

# Evaluation

Evaluation note는 model quality를 정직하게 측정하는 방법을 모읍니다. 핵심은 data를 어떻게 나누고, leakage를 어떻게 찾고, confidence를 어떻게 calibrate하고, training distribution 밖 generalization을 어떻게 검증하는가입니다.

반복되는 원칙은 단순합니다. Metric은 그 뒤의 split과 protocol만큼만 믿을 수 있습니다. 재현되지 않는 reported gain은 modeling 문제가 아니라 evaluation 결함에서 나오는 경우가 많습니다.

기본 estimate는 held-out risk입니다.

$$
\hat{R}(f)
= \frac{1}{m}\sum_{j=1}^{m}
\mathcal{L}(f(x_j), y_j)
$$

이 값은 held-out set이 의도한 generalization claim과 맞을 때만 의미가 있습니다.

## 이동 지도

| 질문 | 시작점 | 이어서 확인할 것 |
| --- | --- | --- |
| score가 어떤 claim을 support하는가? | [Claim-evidence boundary](/concepts/evaluation/claim-evidence-boundary), [Benchmark claim contract](/concepts/evaluation/benchmark-claim-contract) | task, split, metric, baseline |
| evaluation set이 유효한가? | [Evaluation protocol](/concepts/evaluation/evaluation-protocol), [Evaluation set design](/concepts/evaluation/evaluation-set-design) | sampling, exclusion, contamination |
| task에 맞는 metric은 무엇인가? | [Metric](/concepts/evaluation/metric), [Metric selection](/concepts/evaluation/metric-selection) | denominator, threshold, aggregation |
| classification 또는 detection인가? | [Confusion matrix](/concepts/evaluation/confusion-matrix), [Precision and recall](/concepts/evaluation/precision-recall), [Classification metrics](/concepts/evaluation/classification-metrics) | prevalence와 threshold |
| probabilistic prediction인가? | [Probability metrics](/concepts/evaluation/probability-metrics), [Proper scoring rule](/concepts/evaluation/proper-scoring-rule), [Brier score](/concepts/evaluation/brier-score), [Calibration](/concepts/evaluation/calibration) | reliability와 NLL |
| ranking, retrieval, screening 중 무엇인가? | [Ranking metrics](/concepts/evaluation/ranking-metrics), [Negative set](/concepts/evaluation/negative-set) | label completeness와 candidate pool |
| regression 또는 structure인가? | [Regression metrics](/concepts/evaluation/regression-metrics) | unit, target scale, censoring, coordinate mapping |
| generation인가? | [Generation evaluation](/concepts/evaluation/generation-evaluation) | validity, diversity, novelty, task utility |
| improvement를 믿을 수 있는가? | [Baseline](/concepts/evaluation/baseline), [Paired comparison](/concepts/evaluation/paired-comparison), [Confidence interval](/concepts/evaluation/confidence-interval) | seed variance와 effect size |
| leakage 가능성이 있는가? | [Leakage](/concepts/evaluation/leakage), [Test-set contamination](/concepts/evaluation/test-set-contamination), [Train/validation/test split](/concepts/evaluation/train-validation-test-split) | example unit과 split unit |

## 방법 묶음

| 그룹 | 노트 |
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
