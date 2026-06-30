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

## Evaluation Contract

Evaluation note는 score 하나가 아니라 claim을 지지하는 증거 계약입니다.

$$
\mathcal{E}
=
(c,\ \mathcal{T},\ \mathcal{D}_{\mathrm{eval}},\ \pi_{\mathrm{select}},\ M,\ U,\ B)
$$

| Part | Meaning | Typical question |
| --- | --- | --- |
| $c$ | claim | 무엇이 더 낫거나 충분하다고 말하는가? |
| $\mathcal{T}$ | task contract | input, output, allowed context, invalid output은 무엇인가? |
| $\mathcal{D}_{\mathrm{eval}}$ | evaluation set | 어떤 population, split unit, exclusion rule을 대표하는가? |
| $\pi_{\mathrm{select}}$ | selection rule | checkpoint, threshold, prompt, sampler, hyperparameter를 어떻게 골랐는가? |
| $M$ | metric | error, utility, calibration, ranking, generation quality 중 무엇을 측정하는가? |
| $U$ | uncertainty | confidence interval, seed variance, paired comparison, subgroup variance가 있는가? |
| $B$ | baseline | 비교 대상이 같은 data, budget, allowed information을 쓰는가? |

This is the practical rule:

$$
\text{score}
\not\Rightarrow
\text{claim}
\quad
\text{unless task, split, protocol, uncertainty, and baseline are explicit}
$$

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

## Evaluation Stack

Evaluation은 metric 하나가 아니라 stack입니다.

$$
\text{claim}
\leftarrow
\text{task}
\leftarrow
\text{data and split}
\leftarrow
\text{selection rule}
\leftarrow
\text{metric}
\leftarrow
\text{uncertainty}
$$

각 층이 빠지면 score는 좁은 observation일 뿐입니다.

| Layer | Defines | Common Failure |
| --- | --- | --- |
| Claim | 결과가 무엇을 말할 수 있는가 | benchmark score를 general ability로 과장 |
| Task | input, output, allowed context | 다른 task를 같은 metric으로 비교 |
| Data and split | 어떤 population과 holdout을 보는가 | random split으로 OOD claim |
| Selection rule | checkpoint, threshold, prompt, sampler를 어떻게 고르는가 | test feedback으로 best setting 선택 |
| Metric | 어떤 error or utility를 숫자로 만드는가 | decision cost와 metric 불일치 |
| Uncertainty | score 차이가 얼마나 안정적인가 | seed variance 안의 차이를 improvement로 해석 |

## Claim-First Workflow

논문, 프로젝트, benchmark note를 읽을 때는 아래 순서가 가장 안전합니다.

1. Claim을 한 문장으로 줄입니다.
2. Example unit과 split unit을 고정합니다.
3. Final test 전에 고른 selection rule을 찾습니다.
4. Primary metric과 diagnostic metric을 분리합니다.
5. Baseline이 같은 data, budget, allowed information을 쓰는지 확인합니다.
6. Confidence interval, seed variance, paired comparison, subgroup variance 중 하나로 안정성을 봅니다.

이 흐름은 [[ai/evaluation|AI Evaluation]]의 domain-neutral 버전입니다. Molecule, protein, docking, assay처럼 object identity가 중요하면 [[molecular-modeling/data-evaluation|Computational Biology data and evaluation]]에서 split과 leakage를 다시 봅니다.

## Metric Is Not Protocol

Metric은 protocol 안에서만 의미가 있습니다.

$$
\text{reported score}
=
M(
f_{\theta^\*},
\mathcal{D}_{\mathrm{test}},
\pi_{\mathrm{eval}}
)
$$

$\theta^\*$는 선택된 model/checkpoint/prompt/threshold이고, $\pi_{\mathrm{eval}}$은 preprocessing, filtering, denominator, aggregation, invalid-output policy입니다. 같은 metric 이름이라도 $\pi_{\mathrm{eval}}$이 다르면 같은 evidence가 아닙니다.

| Same metric, different protocol | Why comparison breaks |
| --- | --- |
| different denominator | failed examples may be hidden |
| different threshold rule | validation and test evidence are mixed |
| different candidate pool | ranking or screening task changes |
| different preprocessing | feature availability and leakage differ |
| different split unit | generalization claim changes |
| different budget | score may reflect data, compute, or search budget |

## Split Unit Before Metric

Evaluation은 먼저 어떤 unit이 train/test를 가르면 안 되는지 정해야 합니다.

| Domain | Example unit | Split unit to consider |
| --- | --- | --- |
| image classification | image | source, subject, time, class subgroup |
| text or LLM task | prompt/task instance | dataset source, template, benchmark, time |
| retrieval | query-document pair | query, document source, corpus time |
| molecule property | molecule record | scaffold, assay source, time |
| protein task | sequence or structure | family, homolog cluster, template/source |
| docking | protein-ligand complex | ligand scaffold, protein family, complex pair |
| agent task | task trace | environment, tool set, workflow template |

Wrong split unit makes even a clean metric misleading.

## Decision-Oriented Metrics

Metric은 최종 decision과 맞아야 합니다. 같은 model output이라도 decision이 다르면 primary metric도 달라집니다.

| Decision | Prefer | Watch |
| --- | --- | --- |
| classify all examples | accuracy, macro/micro F1, calibration | class imbalance and threshold policy |
| find rare positives | PR-AUC, recall at fixed precision, enrichment | ROC-AUC can look good under imbalance |
| rank candidates | NDCG, MAP, top-k success, enrichment | incomplete labels and candidate pool definition |
| estimate continuous value | RMSE, MAE, Spearman/Pearson, calibration | unit transform and within-series ranking |
| produce probabilities | NLL, Brier score, reliability diagram | probability quality differs from hard accuracy |
| generate samples | validity, novelty, diversity, utility, cost | filtering and attempted denominator |
| decide when not to answer | coverage-risk, selective prediction, conformal coverage | abstention threshold selected on test |

When the decision has asymmetric cost, include the decision rule:

$$
\hat{a}
=
\delta(\hat{p}, \tau, C)
$$

where $\tau$ is the chosen threshold and $C$ is the cost or constraint model.

## Common Traps

| Trap | Better Check |
| --- | --- |
| aggregate score only | subgroup, strata, and failure mode analysis |
| best seed only | seed variance and selection rule |
| leaderboard rank | benchmark saturation and allowed information |
| random split by default | split unit and claimed deployment shift |
| invalid samples removed | attempted denominator and failure taxonomy |
| baseline weaker by construction | simple baseline, matched budget, matched data |
| many metrics searched | multiple comparisons or preregistered primary metric |

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
