---
title: Evaluation
tags:
  - ai
  - evaluation
---

# Evaluation

평가는 모델 목록을 실제 지식으로 바꾸는 기준입니다. AI note를 쓸 때는 어떤 benchmark에서 좋았는지보다, 어떤 split과 metric이 무엇을 검증하는지 먼저 봅니다.

일반화 성능은 보지 못한 데이터 분포에서의 기대 손실로 보는 것이 기본입니다.

$$
R(f)
= \mathbb{E}_{(x,y)\sim p_{\mathrm{test}}}
\left[\mathcal{L}(f(x), y)\right]
$$

실험에서는 이를 finite test set 평균으로 추정합니다.

$$
\hat{R}(f)
= \frac{1}{m}\sum_{j=1}^{m}
\mathcal{L}(f(x_j), y_j)
$$

중요한 점은 $p_{\mathrm{test}}$가 실제로 알고 싶은 deployment distribution을 닮아야 한다는 것입니다.

## Route Map

| Route | Use for | Start |
| --- | --- | --- |
| Task and protocol | 무엇을 예측하고, 무엇을 hold out하며, 어떤 decision을 검증하는가 | [Tasks](/concepts/tasks), [Evaluation protocol](/concepts/evaluation/evaluation-protocol), [Evaluation set design](/concepts/evaluation/evaluation-set-design) |
| Claim boundary | score가 stated conclusion을 지지하는가 | [Claim-evidence boundary](/concepts/evaluation/claim-evidence-boundary), [Benchmark claim contract](/concepts/evaluation/benchmark-claim-contract), [Benchmark intake](/concepts/data/benchmark-intake) |
| Metric choice | classification, regression, ranking, generation, coordinate, probability quality | [Metric](/concepts/evaluation/metric), [Metric selection](/concepts/evaluation/metric-selection), [Generation evaluation](/concepts/evaluation/generation-evaluation) |
| Statistical evidence | confidence interval, seed variance, bootstrap, paired comparison, multiple comparison | [Confidence interval](/concepts/evaluation/confidence-interval), [Seed variance](/concepts/evaluation/seed-variance), [Paired comparison](/concepts/evaluation/paired-comparison) |
| Split and contamination | validation/test separation, pretraining overlap, retrieval contamination, benchmark saturation | [Train/validation/test split](/concepts/evaluation/train-validation-test-split), [Leakage](/concepts/evaluation/leakage), [Test-set contamination](/concepts/evaluation/test-set-contamination) |
| Reliability and failure | calibration, uncertainty, OOD, robustness, error slicing, interpretability | [Calibration](/concepts/evaluation/calibration), [Uncertainty estimation](/concepts/evaluation/uncertainty-estimation), [Error analysis](/concepts/evaluation/error-analysis) |
| Domain-specific splits | scaffold, protein-family, structure-source, assay/source, representation transfer | [Scaffold split](/concepts/evaluation/scaffold-split), [Protein family split](/concepts/evaluation/protein-family-split), [Representation evaluation](/concepts/learning/representation-evaluation) |

## Claim to Evidence Map

Reported result를 볼 때는 먼저 [[concepts/evaluation/claim-evidence-boundary|Claim-evidence boundary]]로 claim 범위를 좁힙니다.

| Claim | 필요한 evidence | 흔한 실패 |
| --- | --- | --- |
| Model이 training task를 fitting한다 | train loss와 diagnostic | optimization success를 generalization으로 착각함 |
| Model이 IID로 generalize한다 | 같은 distribution에서 분리한 held-out validation/test | model selection 중 test set을 건드림 |
| Model이 OOD에 robust하다 | explicit shifted evaluation set | random split이 claimed shift를 검증하지 못함 |
| Model이 calibrated되어 있다 | probability quality metric과 reliability curve | 정확한 label prediction을 calibrated probability로 착각함 |
| Model이 baseline보다 좋다 | relevant baseline과의 paired comparison | 서로 다른 example 또는 selection rule로 비교함 |
| Benchmark score가 broad claim을 지지한다 | benchmark claim contract와 claim-evidence boundary | leaderboard score를 general superiority로 과장함 |
| Method가 design choice를 설명한다 | fixed protocol 아래 ablation | ablation에서 여러 variable을 동시에 바꿈 |
| Selected model이 best다 | model-selection record와 untouched final test | best seed, threshold, prompt, checkpoint를 test feedback으로 고름 |

## Metric Families

| Output type | Primary metric | Diagnostic |
| --- | --- | --- |
| Class label | accuracy, F1, AUROC, PR-AUC | confusion matrix, threshold sweep, calibration |
| Probability | NLL, Brier score, ECE | reliability diagram, selective prediction |
| Scalar regression | RMSE, MAE, $R^2$ | residual plot, rank correlation, interval coverage |
| Ranking | NDCG, MAP, enrichment, top-k success | per-query error, early enrichment, tie policy |
| Generated sample | validity, diversity, novelty, utility | constraint satisfaction, duplicate rate, failure taxonomy |
| Coordinates / structure | RMSD, lDDT-style quality, clash/geometry checks | symmetry correction, atom mapping, interaction quality |
| Calibrated decision | NLL, Brier, ECE, selective risk | reliability diagram, subgroup calibration |

## Domain Routes

| Domain | Start |
| --- | --- |
| Retrieval / QA / ranking | [Retrieval](/concepts/tasks/retrieval), [Question answering](/concepts/tasks/question-answering), [Ranking metrics](/concepts/evaluation/ranking-metrics) |
| Classification / regression | [Classification metrics](/concepts/evaluation/classification-metrics), [Regression metrics](/concepts/evaluation/regression-metrics) |
| Generation | [Sequence generation](/concepts/tasks/sequence-generation), [Generation evaluation](/concepts/evaluation/generation-evaluation) |
| Vision | [Object detection](/concepts/tasks/object-detection), [Localization](/concepts/tasks/localization), [Segmentation](/concepts/tasks/segmentation) |
| Molecule / protein / structure | [Computational Biology](/molecular-modeling), [Virtual screening](/concepts/sbdd/virtual-screening), [Pose quality](/concepts/sbdd/pose-quality), [PoseBusters](/papers/sbdd/posebusters) |
| Agents | [Agent evaluation](/agents/verification/agent-evaluation), [Verification loop](/agents/verification/verification-loop) |
| Representation learning | [Representation evaluation](/concepts/learning/representation-evaluation), [Linear probing](/concepts/learning/linear-probing), [Fine-tuning protocol](/concepts/learning/fine-tuning-protocol) |

## 읽을 때 볼 질문

- train/test split이 실제 generalization을 요구하는가?
- evaluation set 자체가 어떤 population, strata, failure mode를 대표하도록 설계됐는가?
- evaluation protocol이 metric, split, model selection, final test를 분리하는가?
- final test set이 pretraining, retrieval, prompt, feature engineering, leaderboard iteration에 오염되지 않았는가?
- benchmark가 이미 saturated라서 score 차이가 metric noise보다 작은 것은 아닌가?
- metric이 task objective와 맞는가?
- primary metric과 diagnostic metric이 분리되어 있는가?
- failure mode가 wrong output, invalid output, miscalibration, OOD, system failure처럼 분해되어 있는가?
- 결과 차이가 confidence interval이나 seed variance보다 충분히 큰가?
- 같은 test example 위에서 비교했는가, 아니면 독립 aggregate score만 비교했는가?
- 많은 seed, prompt, threshold, ablation 중 가장 좋아 보이는 결과만 고른 것은 아닌가?
- confidence, probability quality, calibration이 필요한 application인가?
- uncertainty, abstention, robustness, interpretability 중 어떤 진단이 필요한가?
- 실패를 data, model, optimization, evaluation 중 어디 문제로 분해할 수 있는가?

## Evidence Record

좋은 evaluation note는 claim을 재구성할 수 있을 만큼 충분한 정보를 보존해야 합니다.

$$
(\text{task},\ \text{split},\ \text{selection rule},\ \text{metric},\ \text{baseline},\ \text{uncertainty})
\rightarrow
\text{claim}
$$

어떤 항목이라도 빠져 있으면 그 score는 stable conclusion이 아니라 incomplete observation으로 취급합니다.

## Failure Mode Decomposition

평가에서 낮은 점수나 큰 개선은 바로 모델 품질로 해석하지 않습니다. 먼저 실패 원인을 분해합니다.

| Failure source | Examples |
| --- | --- |
| Data | label noise, duplicate examples, missing metadata, biased negative set |
| Representation | wrong tokenization, graph construction error, coordinate frame mismatch |
| Optimization | undertraining, instability, bad learning rate, checkpoint selection |
| Model | insufficient inductive bias, over-smoothing, context truncation |
| Metric | wrong threshold, saturated benchmark, unpaired comparison |
| System | nondeterministic run, failed preprocessing, serving/runtime mismatch |

## Minimum Evidence Package

Paper를 wiki note나 post로 바꿀 때는 결과를 해석 가능하게 만드는 최소 evidence package를 기록합니다.

| Field | 중요한 이유 |
| --- | --- |
| Task and output space | classification, ranking, generation, coordinate prediction을 같은 claim처럼 비교하지 않게 합니다 |
| Dataset and split | population과 generalization boundary를 정의합니다 |
| Model-selection rule | validation decision과 final test evidence를 분리합니다 |
| Baseline | method가 relevant simple alternative를 이기는지 보여줍니다 |
| Primary metric | secondary diagnostic을 보기 전에 success의 의미를 고정합니다 |
| Uncertainty or variance | 작은 score difference를 과해석하지 않게 합니다. initialization, split, prompt, sampler가 다르면 [Seed variance](/concepts/evaluation/seed-variance)를 봅니다 |
| Failure modes | score를 재사용 가능한 lesson으로 바꿉니다 |

If a paper does not provide one of these fields, mark it as `to verify` instead of filling the gap from memory.

## Related

- [[ai/learning-methods|Learning methods]]
- [[molecular-modeling/index|Computational Biology]]
- [[papers/index|Papers]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
