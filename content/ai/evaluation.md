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

| Route | Use For | Start |
| --- | --- | --- |
| Task and protocol | what is predicted, what is held out, what decision is being tested | [Tasks](/concepts/tasks), [Evaluation protocol](/concepts/evaluation/evaluation-protocol), [Evaluation set design](/concepts/evaluation/evaluation-set-design) |
| Claim boundary | whether a score supports the stated conclusion | [Claim-evidence boundary](/concepts/evaluation/claim-evidence-boundary), [Benchmark claim contract](/concepts/evaluation/benchmark-claim-contract), [Benchmark intake](/concepts/data/benchmark-intake) |
| Metric choice | classification, regression, ranking, generation, coordinate, probability quality | [Metric](/concepts/evaluation/metric), [Metric selection](/concepts/evaluation/metric-selection), [Generation evaluation](/concepts/evaluation/generation-evaluation) |
| Statistical evidence | confidence intervals, seed variance, bootstrap, paired comparison, multiple comparisons | [Confidence interval](/concepts/evaluation/confidence-interval), [Seed variance](/concepts/evaluation/seed-variance), [Paired comparison](/concepts/evaluation/paired-comparison) |
| Split and contamination | validation/test separation, pretraining overlap, retrieval contamination, benchmark saturation | [Train/validation/test split](/concepts/evaluation/train-validation-test-split), [Leakage](/concepts/evaluation/leakage), [Test-set contamination](/concepts/evaluation/test-set-contamination) |
| Reliability and failure | calibration, uncertainty, OOD, robustness, error slicing, interpretability | [Calibration](/concepts/evaluation/calibration), [Uncertainty estimation](/concepts/evaluation/uncertainty-estimation), [Error analysis](/concepts/evaluation/error-analysis) |
| Domain-specific splits | scaffold, protein-family, structure-source, assay/source, representation transfer | [Scaffold split](/concepts/evaluation/scaffold-split), [Protein family split](/concepts/evaluation/protein-family-split), [Representation evaluation](/concepts/learning/representation-evaluation) |

## Claim to Evidence Map

For any reported result, first narrow the claim with [[concepts/evaluation/claim-evidence-boundary|Claim-evidence boundary]].

| Claim | Evidence Needed | Common Failure |
| --- | --- | --- |
| The model fits the training task | train loss and diagnostics | optimization success is mistaken for generalization |
| The model generalizes IID | held-out validation/test from the same distribution | test set touched during model selection |
| The model is robust OOD | explicit shifted evaluation set | random split does not test the claimed shift |
| The model is calibrated | probability quality metric and reliability curve | accurate labels are treated as calibrated probabilities |
| The model is better than baseline | paired comparison against a relevant baseline | comparison uses different examples or selection rules |
| The benchmark score supports a broad claim | benchmark claim contract and claim-evidence boundary | leaderboard score is overstated as general superiority |
| The method explains a design choice | ablation under fixed protocol | ablation changes multiple variables at once |
| The selected model is best | model-selection record and untouched final test | best seed, threshold, prompt, or checkpoint is selected on test feedback |

## Metric Families

| Output Type | Primary Metrics | Diagnostics |
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

Good evaluation notes should preserve enough information to reconstruct the claim:

$$
(\text{task},\ \text{split},\ \text{selection rule},\ \text{metric},\ \text{baseline},\ \text{uncertainty})
\rightarrow
\text{claim}
$$

If any part is missing, the score should be treated as an incomplete observation rather than a stable conclusion.

## Minimum Evidence Package

When turning a paper into a wiki note or post, record the smallest evidence package that makes the result interpretable.

| Field | Why It Matters |
| --- | --- |
| Task and output space | prevents comparing classification, ranking, generation, and coordinate prediction as if they were the same claim |
| Dataset and split | defines the population and generalization boundary |
| Model-selection rule | separates validation decisions from final test evidence |
| Baseline | shows whether the method beats a relevant simple alternative |
| Primary metric | states what success means before looking at secondary diagnostics |
| Uncertainty or variance | prevents overreading small score differences; use [Seed variance](/concepts/evaluation/seed-variance) when runs differ by initialization, split, prompt, or sampler |
| Failure modes | turns a score into a reusable lesson |

If a paper does not provide one of these fields, mark it as `to verify` instead of filling the gap from memory.

## Related

- [[ai/learning-methods|Learning methods]]
- [[molecular-modeling/index|Computational Biology]]
- [[papers/index|Papers]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
