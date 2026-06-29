---
title: Evaluation Math
tags:
  - math
  - evaluation
---

# Evaluation Math

Evaluation math는 model performance claim을 noise, leakage, calibration problem, benchmark artifact와 분리합니다.

$$
R(f)
=
\mathbb{E}_{(x,y)\sim p_{\mathrm{test}}}
[\mathcal{L}(f(x), y)]
$$

Empirical test score는 이 target의 estimate일 뿐입니다.

## Route Map

| Route | Use for | Start |
| --- | --- | --- |
| Metric definition | 어떤 quantity를 average, rank, threshold, calibrate하는가 | [Metric](/concepts/evaluation/metric), [Metric selection](/concepts/evaluation/metric-selection) |
| Benchmark boundary | benchmark가 어떤 population과 claim을 대표하는가 | [Benchmark intake](/concepts/data/benchmark-intake), [Claim-evidence boundary](/concepts/evaluation/claim-evidence-boundary) |
| Uncertainty | confidence interval, bootstrap estimate, seed/sampler variance | [Confidence interval](/concepts/evaluation/confidence-interval), [Bootstrap evaluation](/concepts/evaluation/bootstrap-evaluation) |
| Comparison | effect size, paired test, multiple-comparison risk | [Effect size](/concepts/evaluation/effect-size), [Paired comparison](/concepts/evaluation/paired-comparison), [Multiple comparisons](/concepts/evaluation/multiple-comparisons) |
| Probability quality | calibrated probability, reliability, proper scoring rule | [Calibration](/concepts/evaluation/calibration), [Brier score](/concepts/evaluation/brier-score), [Probability metrics](/concepts/evaluation/probability-metrics) |
| Statistical claim | reported difference가 noise나 selection bias보다 큰가 | [Statistical significance](/concepts/evaluation/statistical-significance) |

## Estimate and Uncertainty

크기 $m$의 test set에서 empirical risk는 아래와 같습니다.

$$
\hat{R}
=
\frac{1}{m}\sum_{j=1}^{m}
\ell_j
$$

여기서 $\ell_j=\mathcal{L}(f(x_j),y_j)$는 per-example loss입니다.

Approximate standard error는 아래와 같습니다.

$$
\operatorname{SE}(\hat{R})
\approx
\frac{s_\ell}{\sqrt{m}}
$$

여기서 $s_\ell$은 per-example loss의 sample standard deviation입니다. Example이 dependent하거나 heavily stratified되어 있거나 model tuning 이후 선택된 경우 이 approximation은 약해집니다.

## Comparison Map

| Comparison | Preferred evidence | Risk |
| --- | --- | --- |
| 같은 example, 두 model | paired difference와 confidence interval | aggregate score가 per-example dependence를 숨김 |
| 많은 seed | mean, variance, selection rule | best seed를 expected performance로 착각함 |
| 많은 prompt/checkpoint | held-out selection protocol | repeated trial이 false discovery를 키움 |
| 많은 dataset | per-dataset effect size | average score가 중요한 strata의 failure를 숨김 |
| Imbalanced classification | PR-AUC, enrichment, calibrated threshold | severe imbalance에서 ROC-AUC가 좋아 보일 수 있음 |

Paired comparison에서는:

$$
\Delta
=
\frac{1}{m}\sum_{j=1}^{m}
(s_{A,j}-s_{B,j})
$$

여기서 $s_{A,j}$와 $s_{B,j}$는 같은 example 위 두 system의 score 또는 loss입니다.

## Probability Quality

Probability-valued output은 accuracy와 별도의 check가 필요합니다. Classifier가 accurate하더라도 badly calibrated일 수 있고, 낮은 expected loss도 shifted domain에서 unreliable confidence를 숨길 수 있습니다.

| Question | Start |
| --- | --- |
| predicted probability가 empirical frequency와 맞는가? | [Calibration](/concepts/evaluation/calibration), [Reliability diagram](/concepts/evaluation/reliability-diagram) |
| probability score가 proper한가? | [Brier score](/concepts/evaluation/brier-score), [Probability metrics](/concepts/evaluation/probability-metrics) |
| uncertainty가 decision의 일부인가? | [Uncertainty estimation](/concepts/evaluation/uncertainty-estimation), [Conformal prediction](/concepts/evaluation/conformal-prediction) |

## Reporting Checks

Quantity가 고정된 뒤에만 evaluation formula를 사용합니다.

| Question | Mathematical object |
| --- | --- |
| 무엇을 average하는가? | per-example loss, score, binary success, rank statistic, generated-sample property |
| 어떤 set 위에서 계산하는가? | test example, query, target, scaffold, protein, prompt, seed, generated sample |
| 무엇이 random인가? | data sampling, model initialization, decoding, environment, assay noise, bootstrap resampling |
| 무엇을 compare하는가? | paired example, independent aggregate, confidence interval, effect size |
| 무엇을 select했는가? | checkpoint, prompt, threshold, hyperparameter, preprocessing rule, model family |

Reported score $S$에 대해, 유용한 paper note는 아래를 식별해야 합니다.

$$
S
=
\operatorname{aggregate}
\left(
\{s_j\}_{j=1}^{m};
\text{metric},\ \text{selection rule},\ \text{split}
\right)
$$

여기서 $s_j$는 per-unit score이고 aggregation rule은 claim의 일부입니다.

## Checks

- reported score가 point estimate인가, uncertainty-aware comparison인가?
- test set이 model selection과 independent한가?
- comparison이 같은 example 위에서 paired되어 있는가?
- metric이 real decision과 aligned되어 있는가?
- confidence, calibration, abstention이 필요한가?

## Related

- [[math/index|Math]]
- [[ai/evaluation|Evaluation]]
- [[molecular-modeling/data-evaluation|Computational Biology data and evaluation]]
