---
title: AI
tags:
  - ai
---

# AI

AI 전반을 정리하는 입구입니다. 이 페이지는 공개 블로그 표면에 가까운 안내 페이지이고, 세부 개념은 영어 wiki 노트로 연결합니다.

이 페이지는 한글 안내 페이지입니다. 링크된 `concepts/*` 문서는 재사용 가능한 canonical wiki note로 영어를 유지합니다.

여기서 다루려는 핵심은 특정 모델 이름을 외우는 것이 아니라, 모델이 어떤 구조로 정보를 처리하고, 어떤 학습 신호로 표현을 만들고, 어떤 방식으로 생성하거나 판단하는지입니다.

## 큰 축

- Math Foundations: probability, linear algebra, likelihood, information theory 같은 공통 수식 기반
- Machine Learning: 예측 문제, feature, loss, regularization, validation을 다루는 기본 층
- Data: dataset, annotation, sampling, benchmark, provenance를 다루는 층
- Modality: text, image, video, audio, molecular/protein structure처럼 입력과 출력 신호가 어떤 형태인지 보는 층
- Task: classification, retrieval, detection, segmentation, question answering처럼 모델이 무엇을 출력해야 하는지 보는 층
- Architecture: 모델이 정보를 흘려보내는 구조
- Learning: 어떤 supervision이나 objective로 표현을 학습하는지
- Generation: 데이터를 만들거나 변환하는 방식
- Systems: training run, inference, serving, reproducibility처럼 모델을 실제로 돌리는 방식
- Evaluation: 모델이 실제로 일반화했는지 확인하는 방식
- Geometry and scientific AI: graph, coordinate, protein, molecule 같은 구조적 입력을 다루는 방식

## Machine Learning

Machine learning은 AI 노트의 기본 층입니다. 딥러닝 모델을 보기 전에도 problem type, feature, loss, optimization, regularization, validation을 구분해야 합니다.

- [[concepts/math/index|Math foundations]]
- [[math/index|Math gateway]]
- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/math/singular-value-decomposition|Singular value decomposition]]
- [[concepts/math/calculus|Calculus]]
- [[concepts/math/matrix-calculus|Matrix calculus]]
- [[concepts/math/random-variable|Random variable]]
- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/statistical-estimator|Statistical estimator]]
- [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]]
- [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[ai/machine-learning|Machine learning gateway]]
- [[concepts/machine-learning/index|Machine learning]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/linear-model|Linear model]]
- [[concepts/machine-learning/tree-based-model|Tree-based model]]
- [[concepts/machine-learning/kernel-method|Kernel method]]
- [[concepts/machine-learning/regularization|Regularization]]

## Data and Benchmarks

AI 모델은 데이터 정의 위에서만 의미가 있습니다. 어떤 example을 모았고, label이 어떻게 만들어졌고, split과 benchmark가 무엇을 검증하는지 모르면 architecture 비교도 흔들립니다.

- [[concepts/data/index|Data]]
- [[entities/dataset|Dataset]]
- [[concepts/data/data-curation|Data curation]]
- [[concepts/data/data-versioning|Data versioning]]
- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/data/benchmark|Benchmark]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]

## Modalities

모델을 보기 전에 입력과 출력의 형태를 먼저 봐야 합니다. text, image, video, audio, molecule, protein structure는 모두 서로 다른 preprocessing, tokenization, leakage 위험, evaluation 기준을 갖습니다.

- [[concepts/modalities/index|Modalities]]
- [[concepts/modalities/text|Text]]
- [[concepts/modalities/sequence|Sequence]]
- [[concepts/modalities/image|Image]]
- [[concepts/modalities/video|Video]]
- [[concepts/modalities/audio|Audio]]
- [[concepts/modalities/tabular|Tabular]]
- [[concepts/modalities/graph|Graph]]
- [[concepts/modalities/3d-structure|3D structure]]
- [[concepts/modalities/modality-alignment|Modality alignment]]
- [[concepts/modalities/missing-modality|Missing modality]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[entities/index|Entities]]

## Tasks and Outputs

Task는 모델의 출력 공간과 평가 기준을 정합니다. 같은 image input이라도 classification, detection, segmentation, captioning은 전혀 다른 문제이고, 같은 text input이라도 retrieval, question answering, sequence generation은 실패 방식이 다릅니다.

- [[concepts/tasks/index|Tasks]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/tasks/captioning|Captioning]]
- [[concepts/tasks/question-answering|Question answering]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/tasks/structured-prediction|Structured prediction]]
- [[concepts/tasks/time-series-forecasting|Time-series forecasting]]
- [[concepts/tasks/anomaly-detection|Anomaly detection]]

## Architectures

아키텍처 노트는 입력의 형태와 inductive bias를 기준으로 봅니다. 이미지, sequence, graph, structure, set처럼 데이터가 달라지면 좋은 기본 구조도 달라집니다.

- [[ai/architectures|아키텍처 지도]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/state-space-model|State-space models]]
- [[concepts/architectures/mixture-of-experts|Mixture of experts]]

## Learning Methods

학습 방법은 label이 충분한 상황과 부족한 상황을 나눠서 봅니다. 특히 [[concepts/learning/self-supervised-learning|self-supervised learning]], [[concepts/learning/jepa|JEPA]], [[concepts/learning/contrastive-learning|contrastive learning]]은 representation을 어떻게 만들 것인가와 직접 연결됩니다.

- [[ai/learning-methods|Learning methods gateway]]
- [[concepts/learning/index|Learning methods]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/preference-optimization|Preference optimization]]

## Generative Models

생성 모델은 likelihood, denoising, flow, autoregressive factorization처럼 서로 다른 관점에서 볼 수 있습니다. Bio-AI에서는 molecule generation, protein design, structure generation과 연결됩니다.

- [[ai/generative-models|Generative models gateway]]
- [[concepts/generative-models/index|Generative models]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]

## Systems and Operations

모델은 학습되고, 저장되고, 실행되고, 평가되는 시스템입니다. 같은 architecture라도 training run 관리, inference path, latency/throughput 목표, reproducibility 기준에 따라 실제 가치는 달라집니다.

- [[concepts/systems/index|AI systems]]
- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/environment-management|Environment management]]
- [[concepts/systems/inference|Inference]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[concepts/systems/storage-io|Storage and IO]]
- [[concepts/systems/observability|Observability]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[infra/index|Infra]]

## Geometry and Scientific AI

AI를 Bio-AI와 연결할 때는 구조적 입력을 어떻게 표현하는지가 중요합니다. 단백질, 분자, pocket, complex는 단순 text token이 아니라 graph와 coordinate를 함께 갖습니다.

- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/protein-modeling/index|Protein modeling concepts]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[bio-ai/index|Bio-AI]]

## LLM and Agents

LLM은 생성 모델이면서 agent workflow의 실행 엔진이 될 수 있습니다. 여기서는 model 자체보다 context, retrieval, verification이 더 중요합니다.

- [[concepts/llm/index|LLM concepts]]
- [[concepts/llm/language-model|Language model]]
- [[concepts/llm/context-window|Context window]]
- [[concepts/llm/prompting|Prompting]]
- [[concepts/llm/decoding|Decoding]]
- [[concepts/llm/structured-output|Structured output]]
- [[concepts/llm/hallucination-grounding|Hallucination and grounding]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[agents/index|Agents]]
- [[agents/core/agent-architecture|Agent architecture]]
- [[agents/workflows/agent-orchestration|Agent orchestration]]

## Evaluation

AI 노트는 평가 기준 없이 모델 목록이 되는 것을 피해야 합니다. 그래서 leakage, split, calibration, out-of-distribution generalization 같은 평가 노트를 계속 연결합니다.

- [[ai/evaluation|Evaluation gateway]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/calibration|Calibration]]

## Related

- [[bio-ai/index|Bio-AI]]
- [[agents/index|Agents]]
- [[ai/generative-models|Generative models]]
- [[ai/learning-methods|Learning methods]]
