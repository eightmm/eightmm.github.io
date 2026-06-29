---
title: AI
tags:
  - ai
---

# AI

AI 전반을 정리하는 입구입니다. 여기서는 모델이 무엇을 입력받고, 어떤 구조로 계산하며, 어떤 신호로 학습하고, 어떤 근거로 평가되는지를 정리합니다.

목표는 모델 이름을 많이 나열하는 것이 아니라, AI 문헌과 구현을 읽을 때 반복해서 나오는 축을 안정적으로 분리하는 것입니다. 도메인 객체 자체는 [[molecular-modeling/index|계산생물학]]에서, 수식과 확률/기하 언어는 [[math/index|수학]]에서, GPU/HPC 같은 실행 환경은 [[infra/index|Infra]]에서 다룹니다.

$$
\text{AI system}
=
\text{data}
+ \text{representation}
+ \text{architecture}
+ \text{objective}
+ \text{evaluation}
+ \text{runtime}
$$

## 먼저 볼 지도

| 영역 | 용도 | 시작점 |
| --- | --- | --- |
| Math | probability, linear algebra, calculus, likelihood, information theory 같은 수학 기반 | [Math](/math) |
| Machine Learning | prediction problem, feature, loss, optimization, validation | [Machine Learning](/ai/machine-learning) |
| Model Reading Map | 논문 한 편을 input, representation, architecture, objective, evaluation, runtime으로 분해 | [Model Reading Map](/ai/model-reading-map) |
| Architectures | MLP, CNN, RNN, Transformer, GNN, SSM/Mamba, MoE 같은 모델 구조 | [Architectures](/ai/architectures) |
| Learning Methods | supervised learning, self-supervised learning, contrastive learning, JEPA, fine-tuning, preference/RL objective | [Learning Methods](/ai/learning-methods) |
| Generative Models | autoregressive model, VAE, GAN, diffusion, score model, flow matching, normalizing flow | [Generative Models](/ai/generative-models) |
| Evaluation | metric, split, leakage, calibration, OOD, uncertainty, failure analysis | [Evaluation](/ai/evaluation) |
| Systems | training run, inference, serving, environment, artifact, reproducibility | [Systems](/ai/systems) |
| Agents | tool use, memory, planning, verification, orchestration | [Agents](/agents) |

## 분류 기준

AI note는 아래 질문으로 위치를 정합니다.

| 질문 | 이동할 곳 |
| --- | --- |
| 필요한 수학 정의인가? | [Math](/math), [Math foundations](/concepts/math) |
| 입력과 출력이 무엇인가? | [Modalities](/concepts/modalities), [Tasks](/concepts/tasks) |
| 논문 한 편을 어떻게 분해해서 읽을 것인가? | [Model Reading Map](/ai/model-reading-map) |
| 예측 문제와 loss의 기본형인가? | [Machine Learning](/ai/machine-learning) |
| 모델 내부 구조인가? | [Architectures](/ai/architectures) |
| supervision, objective, transfer 방식인가? | [Learning Methods](/ai/learning-methods) |
| sample을 만들거나 distribution을 모델링하는가? | [Generative Models](/ai/generative-models) |
| 성능 claim을 어떻게 검증하는가? | [Evaluation](/ai/evaluation) |
| 실행, serving, reproducibility 문제인가? | [Systems](/ai/systems), [Infra](/infra) |
| LLM이 도구를 쓰고 작업을 끝내는 방식인가? | [Agents](/agents) |

## 영역 경계

| 경계 | AI에서 다루는 것 | 다른 곳에서 다루는 것 |
| --- | --- | --- |
| Math | loss, optimizer, architecture 식을 모델 관점에서 읽기 | derivative, likelihood, entropy, group action 자체는 [Math](/math) |
| Computational Biology | protein, molecule, structure 데이터를 모델 입력으로 쓰는 법 | protein, ligand, pocket, assay, split 단위 정의는 [Computational Biology](/molecular-modeling) |
| Infra | training run, inference contract, serving boundary | GPU memory, Slurm, storage, server operation은 [Infra](/infra) |
| Papers | claim type과 방법론을 분류하는 기준 | 개별 논문 요약과 reading status는 [Papers](/papers) |

## 기본 읽기 경로

1. [[math/index|Math]]에서 vector, probability, likelihood, entropy/KL, calculus를 먼저 잡습니다.
2. [[ai/machine-learning|Machine Learning]]에서 data, target, loss, optimization, validation의 기본 구조를 봅니다.
3. [[ai/architectures|Architectures]]에서 입력 구조별 inductive bias를 비교합니다.
4. [[ai/learning-methods|Learning Methods]]에서 label, pretraining signal, transfer, preference objective를 분리합니다.
5. [[ai/generative-models|Generative Models]]에서 likelihood, denoising, score, velocity, sampling 관점을 비교합니다.
6. [[ai/evaluation|Evaluation]]에서 split, metric, leakage, calibration, failure mode를 확인합니다.
7. [[ai/systems|Systems]]에서 training run, inference, serving, environment, reproducibility 경계를 봅니다.

## 우선 채울 층

AI 쪽은 너무 빨리 논문명 중심으로 커지기 쉽습니다. 그래서 먼저 아래 층을 안정적으로 채웁니다.

| 우선순위 | 중요한 이유 | 이동할 곳 |
| --- | --- | --- |
| Basic ML grammar | 모든 모델은 data, target, loss, optimization, evaluation으로 환원됩니다. | [Machine Learning](/ai/machine-learning) |
| Architecture families | 논문 차이는 대부분 어떤 inductive bias를 넣었는지로 설명됩니다. | [Architectures](/ai/architectures) |
| Learning signal | SSL, contrastive, JEPA, preference, RL은 architecture보다 objective 차이가 핵심일 때가 많습니다. | [Learning Methods](/ai/learning-methods) |
| Generative process | 생성 모델은 likelihood, denoising, score, velocity, sampler를 구분해야 합니다. | [Generative Models](/ai/generative-models) |
| Evaluation habit | 모델 주장은 split, metric, selection rule 없이는 안정적인 지식이 아닙니다. | [Evaluation](/ai/evaluation) |
| System boundary | 좋은 모델도 run, artifact, serving, reproducibility가 없으면 재사용하기 어렵습니다. | [Systems](/ai/systems) |

## Note Template

새 AI note는 가능하면 아래 순서를 따릅니다.

| 항목 | 적을 내용 |
| --- | --- |
| Problem | 입력 $x$, 출력 $y$, task unit |
| Representation | token, graph, coordinate, embedding, feature |
| Model | architecture family와 inductive bias |
| Objective | loss, likelihood, reward, score, velocity, contrast |
| Training | data, split, optimizer, selection rule |
| Evaluation | metric, baseline, uncertainty, failure mode |
| Boundary | Math, Computational Biology, Infra, Agents 중 어디와 연결되는지 |

## 입력 대상별 경로

| 입력 | 시작점 | 구조 / 방법 |
| --- | --- | --- |
| Text / sequence | [Text](/concepts/modalities/text), [Sequence](/concepts/modalities/sequence) | [Transformer](/concepts/architectures/transformer), [State-space model](/concepts/architectures/state-space-model) |
| Image / video | [Image](/concepts/modalities/image), [Video](/concepts/modalities/video) | [CNN](/concepts/architectures/cnn), [Vision Transformer](/concepts/architectures/vision-transformer) |
| Graph / set | [Graph](/concepts/modalities/graph) | [GNN](/concepts/architectures/gnn), [Deep Sets](/concepts/architectures/deep-sets) |
| Graph with learned structure | [Representation contract](/concepts/modalities/representation-contract) | [Graph construction](/concepts/architectures/graph-construction), [Graph Transformer](/concepts/architectures/graph-transformer) |
| 3D / geometry | [3D structure](/concepts/modalities/3d-structure), [Geometry](/concepts/math/geometry) | [Geometric deep learning](/concepts/geometric-deep-learning), [Coordinate modeling contract](/concepts/geometric-deep-learning/coordinate-modeling-contract) |
| Molecule / protein | [Computational Biology](/molecular-modeling) | [Molecular modeling](/concepts/molecular-modeling), [Protein modeling](/concepts/protein-modeling), [SBDD concepts](/concepts/sbdd) |
| Agent workflow | [Agents](/agents) | [Core](/agents/core), [Tools](/agents/tools), [Verification](/agents/verification), [Workflows](/agents/workflows) |

## 논문과 포스트를 읽을 때

새 AI 논문이나 글감을 넣을 때는 모델 이름보다 아래 항목을 먼저 고정합니다.

| 먼저 볼 것 | 확인할 내용 | 시작점 |
| --- | --- | --- |
| Input object | text, image, graph, set, coordinate, molecule, protein, agent state 중 무엇인가 | [Modalities](/concepts/modalities), [Tasks](/concepts/tasks) |
| Task output | class, scalar, ranking, sequence, graph, coordinate, sample, action 중 무엇인가 | [Task specification](/concepts/tasks/task-specification), [Task output space](/concepts/tasks/task-output-space) |
| Representation | raw object가 token, graph, coordinate, embedding, conformer로 어떻게 바뀌는가 | [Representation contract](/concepts/modalities/representation-contract) |
| Architecture | 어떤 inductive bias, parameter sharing, complexity를 쓰는가 | [Architectures](/ai/architectures), [Architecture-objective fit](/concepts/architectures/architecture-objective-fit) |
| Learning signal | label, mask, contrast, preference, reward, denoising, velocity 중 무엇인가 | [Learning Methods](/ai/learning-methods), [Objective taxonomy](/concepts/learning/objective-taxonomy) |
| Objective | loss, likelihood, score, reward, metric이 어떻게 정의되는가 | [Machine Learning](/ai/machine-learning), [Math](/math) |
| Selection rule | checkpoint, hyperparameter, threshold, prompt, filter를 어떻게 골랐는가 | [Model selection](/concepts/machine-learning/model-selection) |
| Evaluation claim | 어떤 split, metric, baseline, uncertainty로 주장을 검증하는가 | [Evaluation](/ai/evaluation) |
| Uncertainty and calibration | 결과 차이와 probability claim을 얼마나 믿을 수 있는가 | [Confidence interval](/concepts/evaluation/confidence-interval), [Calibration](/concepts/evaluation/calibration) |
| System boundary | training, inference, serving, reproducibility, tool-use 문제가 있는가 | [Systems](/ai/systems), [Infra](/infra), [Agents](/agents) |

## Related

- [[concepts/index|Concepts]]
- [[ai/model-reading-map|Model Reading Map]]
- [[math/index|Math]]
- [[molecular-modeling/index|Computational Biology]]
- [[infra/index|Infra]]
- [[agents/index|Agents]]
