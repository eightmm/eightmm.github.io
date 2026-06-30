---
title: Tasks
tags:
  - tasks
  - machine-learning
---

# Tasks

Task는 model이 무엇을 출력해야 하고 성공을 어떻게 측정할지 정의합니다. Architecture는 정보가 흐르는 방식을, modality는 input signal을, task는 target behavior를 설명합니다.

같은 input도 여러 task를 가질 수 있습니다.

$$
\hat{y} = f_\theta(x), \qquad \theta^\star = \arg\min_\theta \mathcal{L}(f_\theta(x), y)
$$

중요한 질문은 $y$가 무엇을 뜻하는가입니다. $y$는 class, scalar, ranked list, generated sequence, box, mask, answer, retrieved item, structured object일 수 있습니다.

## 이동 지도

| 필요 | 시작점 | 일반적인 metric 경계 |
| --- | --- | --- |
| define the full task contract | [Task specification](/concepts/tasks/task-specification) | loss, metric, split, validity rule |
| choose the output type | [Task output space](/concepts/tasks/task-output-space) | class, scalar, rank, sequence, graph, coordinate |
| connect input modality to task | [Modality-task map](/concepts/modalities/modality-task-map) | raw input, representation, output, metric |
| predict labels or values | [Classification](/concepts/machine-learning/classification), [Regression](/concepts/machine-learning/regression) | calibration, threshold, error scale |
| rank or retrieve items | [Ranking](/concepts/machine-learning/ranking), [Retrieval](/concepts/tasks/retrieval), [Reranking](/concepts/tasks/reranking) | top-k, NDCG, enrichment, recall |
| generate objects | [Sequence generation](/concepts/tasks/sequence-generation), [Structured prediction](/concepts/tasks/structured-prediction) | validity, diversity, likelihood, downstream score |
| localize in space | [Localization](/concepts/tasks/localization), [Object detection](/concepts/tasks/object-detection), [Segmentation](/concepts/tasks/segmentation) | IoU, mask quality, localization error |
| model molecules or proteins | [Property prediction](/concepts/tasks/property-prediction), [Interaction prediction](/concepts/tasks/interaction-prediction) | assay context, split unit, leakage risk |
| model structure | [Coordinate prediction](/concepts/tasks/coordinate-prediction), [Graph prediction](/concepts/tasks/graph-prediction) | geometry validity, invariance, equivariance |

## Task Contract

Task는 입력과 출력 이름만이 아니라 decision boundary입니다.

$$
\mathcal{T}
=
(\mathcal{X},\ \mathcal{Y},\ \mathcal{C},\ \mathcal{L},\ M,\ \mathcal{S})
$$

where $\mathcal{X}$ is input space, $\mathcal{Y}$ is output space, $\mathcal{C}$ is allowed context, $\mathcal{L}$ is training loss, $M$ is evaluation metric, and $\mathcal{S}$ is the split rule.

| Contract part | Ask | Route |
| --- | --- | --- |
| input space | what object does the model receive after preprocessing? | [Modalities](/concepts/modalities), [Representation contract](/concepts/modalities/representation-contract) |
| output space | what counts as a valid prediction? | [Task output space](/concepts/tasks/task-output-space) |
| allowed context | what information is available at inference time? | [Evaluation protocol](/concepts/evaluation/evaluation-protocol) |
| loss | what signal is optimized during training? | [Loss function](/concepts/machine-learning/loss-function), [Learning methods](/concepts/learning) |
| metric | what behavior decides success? | [Metric selection](/concepts/evaluation/metric-selection) |
| split rule | what unit must be held out? | [Dataset split contract](/concepts/data/dataset-split-contract) |

If two papers use the same model family but different $\mathcal{Y}$, context, metric, or split, they are not solving the same task.

## Output Space Map

| Output type | Typical task | Metric route | Failure to count |
| --- | --- | --- | --- |
| class | classification, detection decision | [Classification metrics](/concepts/evaluation/classification-metrics) | threshold and prevalence mismatch |
| scalar | regression, property prediction | [Regression metrics](/concepts/evaluation/regression-metrics) | unit, censoring, target scale |
| probability | risk, confidence, uncertainty | [Probability metrics](/concepts/evaluation/probability-metrics), [Calibration](/concepts/evaluation/calibration) | miscalibration |
| ranked list | retrieval, screening, reranking | [Ranking metrics](/concepts/evaluation/ranking-metrics) | candidate pool mismatch |
| sequence | text, protein, SMILES, trajectory | [Generation evaluation](/concepts/evaluation/generation-evaluation) | invalid syntax and filtered denominator |
| set or graph | structured prediction, graph prediction | [Metric selection](/concepts/evaluation/metric-selection) | permutation, matching, graph validity |
| coordinates | structure, pose, localization | [Pose quality](/concepts/sbdd/pose-quality), [Regression metrics](/concepts/evaluation/regression-metrics) | alignment, symmetry, atom mapping |
| action or trace | agents, tool workflows | [Agent evaluation](/agents/verification/agent-evaluation) | side effect, recovery, completion evidence |

## Task vs Objective vs Architecture

These are different axes.

| Axis | Example | Common Confusion |
| --- | --- | --- |
| task | classify activity, rank candidates, generate molecule, predict coordinates | described as model family |
| objective | cross-entropy, contrastive loss, denoising score, flow matching | treated as the final task |
| architecture | CNN, Transformer, GNN, SSM, equivariant model | credited for gains caused by objective or data |
| modality | sequence, image, graph, 3D structure | used as if it already defines the output |
| evaluation | AUROC, RMSE, RMSD, NDCG, validity | metric chosen after seeing results |

For example, a Transformer can be used for classification, retrieval, sequence generation, or reranking. A diffusion model can be trained for image generation, molecule generation, structure generation, or denoising representation learning. The task note should pin down the output behavior before architecture or objective claims.

## Generalization Unit

Task specification should name the unit of generalization.

| Task family | Example unit | Split unit risk |
| --- | --- | --- |
| property prediction | molecule, material, protein | scaffold, homolog, assay source |
| interaction prediction | pair or complex | target family, ligand scaffold, pair leakage |
| retrieval | query-candidate pair | shared document, shared query template, corpus time |
| generation | sampled object | duplicated training object, invalid filtered output |
| structure prediction | sequence, complex, residue set | template, homolog, structure source |
| agent task | task instance and tool trace | repeated workflow template or hidden tool state |

The split unit should be stronger than the claim. If the claim is transfer to new targets, a random row split is usually too weak.

## Task 묶음

| 묶음 | 노트 |
| --- | --- |
| Core ML | [Classification](/concepts/machine-learning/classification), [Regression](/concepts/machine-learning/regression), [Ranking](/concepts/machine-learning/ranking) |
| Search | [Retrieval](/concepts/tasks/retrieval), [Similarity search](/concepts/tasks/similarity-search), [Reranking](/concepts/tasks/reranking) |
| Vision and spatial | [Object detection](/concepts/tasks/object-detection), [Localization](/concepts/tasks/localization), [Segmentation](/concepts/tasks/segmentation) |
| Language | [Captioning](/concepts/tasks/captioning), [Question answering](/concepts/tasks/question-answering), [Sequence generation](/concepts/tasks/sequence-generation), [Sequence-to-sequence](/concepts/tasks/sequence-to-sequence) |
| Structured outputs | [Structured prediction](/concepts/tasks/structured-prediction), [Coordinate prediction](/concepts/tasks/coordinate-prediction), [Graph prediction](/concepts/tasks/graph-prediction) |
| Time and monitoring | [Time-series forecasting](/concepts/tasks/time-series-forecasting), [Anomaly detection](/concepts/tasks/anomaly-detection) |

## Paper Reading Pattern

When reading a method paper, rewrite the task before reading the model diagram.

| Field | Write |
| --- | --- |
| input | raw object and model-ready representation |
| output | exact prediction type and validity rule |
| context | what information is allowed at inference |
| loss | training signal and supervision source |
| metric | primary metric and diagnostics |
| split | held-out unit and leakage audit |
| baseline | simplest relevant task baseline |

This keeps model novelty separate from task definition.

## 확인할 것

- output space는 무엇인가?
- 어떤 [[concepts/tasks/task-output-space|task output space]]가 valid prediction을 제한하는가?
- input, output, validity, loss, metric, split을 포함한 task specification이 있는가?
- raw input, representation, output space, metric을 연결하는 modality-task map이 있는가?
- [[concepts/data/example-unit|example unit]]은 무엇인가?
- 어떤 [[concepts/data/split-unit|split unit]]이 의도한 generalization을 검증하는가?
- model은 predicting, ranking, retrieving, generating, localizing, segmenting 중 무엇을 하는가?
- retrieval은 최종 output인가, similarity-search stage인가, reranking pipeline인가?
- task는 entity-level, pairwise, context-conditioned, structured 중 무엇인가?
- output은 independent, sequential, structured, ranked, spatial, temporal 중 무엇인가?
- metric이 user-facing behavior와 맞는가?
- data split이 이 task의 leakage를 막는가?
- output이 valid object, syntax, molecule, structure, action으로 제한되는가?

## Related

- [[ai/index|AI]]
- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/tasks/task-output-space|Task output space]]
- [[concepts/modalities/index|Modalities]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/architectures/index|Architectures]]
- [[concepts/learning/index|Learning methods]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
