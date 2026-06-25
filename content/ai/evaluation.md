---
title: Evaluation
tags:
  - ai
  - evaluation
---

# Evaluation

평가는 모델 목록을 실제 지식으로 바꾸는 기준입니다. AI note를 쓸 때는 어떤 benchmark에서 좋았는지보다, 어떤 split과 metric이 무엇을 검증하는지 먼저 봅니다.

이 페이지는 한글 안내 페이지입니다. 링크된 `concepts/evaluation/*` 문서는 영어 canonical wiki note로 유지합니다.

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

## 핵심 노트

- [[concepts/tasks/index|Tasks]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/failure-mode-taxonomy|Failure mode taxonomy]]
- [[concepts/evaluation/confusion-matrix|Confusion matrix]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/evaluation/proper-scoring-rule|Proper scoring rule]]
- [[concepts/evaluation/brier-score|Brier score]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/bootstrap-evaluation|Bootstrap evaluation]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/cross-validation|Cross-validation]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/reliability-diagram|Reliability diagram]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/conformal-prediction|Conformal prediction]]
- [[concepts/evaluation/selective-prediction|Selective prediction]]
- [[concepts/evaluation/robustness|Robustness]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[concepts/evaluation/interpretability|Interpretability]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/learning/linear-probing|Linear probing]]
- [[concepts/learning/fine-tuning-protocol|Fine-tuning protocol]]

## 분야별 평가 연결

- Retrieval/QA: [[concepts/tasks/retrieval|Retrieval]], [[concepts/tasks/question-answering|Question answering]]
- Classification/regression: [[concepts/evaluation/classification-metrics|Classification metrics]], [[concepts/evaluation/regression-metrics|Regression metrics]]
- Generation: [[concepts/tasks/sequence-generation|Sequence generation]], [[concepts/evaluation/generation-evaluation|Generation evaluation]]
- Vision: [[concepts/tasks/object-detection|Object detection]], [[concepts/tasks/segmentation|Segmentation]]
- Molecule: [[concepts/evaluation/scaffold-split|Scaffold split]], [[concepts/sbdd/virtual-screening|Virtual screening]]
- Protein: [[concepts/evaluation/protein-family-split|Protein family split]], [[concepts/protein-modeling/sequence-structure-alignment|Sequence-structure alignment]]
- Structure: [[concepts/sbdd/pose-quality|Pose quality]], [[papers/sbdd/posebusters|PoseBusters]]
- Agent: [[agents/verification/agent-evaluation|Agent evaluation]], [[agents/verification/verification-loop|Verification loop]]
- Ranking: [[concepts/evaluation/ranking-metrics|Ranking metrics]], [[concepts/tasks/retrieval|Retrieval]]
- Representation learning: [[concepts/learning/representation-evaluation|Representation evaluation]], [[concepts/learning/linear-probing|Linear probing]], [[concepts/learning/fine-tuning-protocol|Fine-tuning protocol]]

## 읽을 때 볼 질문

- train/test split이 실제 generalization을 요구하는가?
- evaluation protocol이 metric, split, model selection, final test를 분리하는가?
- metric이 task objective와 맞는가?
- primary metric과 diagnostic metric이 분리되어 있는가?
- failure mode가 wrong output, invalid output, miscalibration, OOD, system failure처럼 분해되어 있는가?
- 결과 차이가 confidence interval이나 seed variance보다 충분히 큰가?
- confidence, probability quality, calibration이 필요한 application인가?
- uncertainty, abstention, robustness, interpretability 중 어떤 진단이 필요한가?
- 실패를 data, model, optimization, evaluation 중 어디 문제로 분해할 수 있는가?

## Related

- [[ai/learning-methods|Learning methods]]
- [[bio-ai/index|Bio-AI]]
- [[papers/index|Papers]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
