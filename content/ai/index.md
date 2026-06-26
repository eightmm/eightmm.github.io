---
title: AI
tags:
  - ai
---

# AI

AI 전반을 정리하는 입구입니다.

여기서 목표는 모델 이름을 많이 나열하는 것이 아니라, AI 문헌과 구현을 읽을 때 반복해서 나오는 축을 안정적으로 분리하는 것입니다.

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

- **Math**: probability, linear algebra, calculus, likelihood, information theory. [Start](/math)
- **Machine Learning**: prediction problem, feature, loss, optimization, validation. [Start](/ai/machine-learning)
- **Architectures**: MLP, CNN, RNN, Transformer, GNN, SSM/Mamba, MoE. [Start](/ai/architectures)
- **Learning Methods**: supervised learning, self-supervised learning, contrastive learning, JEPA, fine-tuning, preference/RL-style objective. [Start](/ai/learning-methods)
- **Generative Models**: autoregressive model, VAE, GAN, diffusion, score model, flow matching, normalizing flow. [Start](/ai/generative-models)
- **Evaluation**: metric, split, leakage, calibration, OOD, uncertainty, failure analysis. [Start](/ai/evaluation)
- **Agents**: tool use, memory, planning, verification, orchestration. [Start](/agents)
- **Paper Intake**: input/output, architecture, objective, evidence, and system boundary. [Start](/ai/paper-intake)
- **Paper Claim Patterns**: architecture, learning method, generative model, evaluation, scaling, systems, and agent claim routing. [Start](/ai/paper-claim-patterns)
- **Post Intake**: Korean synthesis posts that combine AI, computational biology, and Math. [Start](/posts/ai-molecular-math-post-intake)
- **Cross-Axis Contract**: object, representation, model, objective, evidence, and public boundary for mixed AI + computational biology + Math claims. [Start](/concepts/ai-computational-biology-math-contract)
- **Coverage Matrix**: check whether a topic has object, data, model, objective, evidence, and public boundary notes. [Start](/concepts/coverage-matrix)

## 분류 기준

AI note는 아래 질문으로 위치를 정합니다.

- 필요한 수학 정의인가? → [Math](/math), [Math foundations](/concepts/math)
- 입력과 출력이 무엇인가? → [Modalities](/concepts/modalities), [Tasks](/concepts/tasks)
- 예측 문제와 loss의 기본형인가? → [Machine Learning](/ai/machine-learning)
- 모델 내부 구조인가? → [Architectures](/ai/architectures)
- supervision, objective, transfer 방식인가? → [Learning Methods](/ai/learning-methods)
- sample을 만들거나 distribution을 모델링하는가? → [Generative Models](/ai/generative-models)
- 성능 claim을 어떻게 검증하는가? → [Evaluation](/ai/evaluation)
- 실행, serving, reproducibility 문제인가? → [Systems](/concepts/systems), [Infra](/infra)
- LLM이 도구를 쓰고 작업을 끝내는 방식인가? → [Agents](/agents)

## 기본 읽기 경로

1. [[math/index|Math]]에서 vector, probability, likelihood, entropy/KL, calculus를 먼저 잡습니다.
2. [[ai/machine-learning|Machine Learning]]에서 data, target, loss, optimization, validation의 기본 구조를 봅니다.
3. [[ai/architectures|Architectures]]에서 입력 구조별 inductive bias를 비교합니다.
4. [[ai/learning-methods|Learning Methods]]에서 label, pretraining signal, transfer, preference objective를 분리합니다.
5. [[ai/generative-models|Generative Models]]에서 likelihood, denoising, score, velocity, sampling 관점을 비교합니다.
6. [[ai/evaluation|Evaluation]]에서 split, metric, leakage, calibration, failure mode를 확인합니다.

## 입력 대상별 경로

| Input | Start | Architecture / Method |
| --- | --- | --- |
| Text / sequence | [Text](/concepts/modalities/text), [Sequence](/concepts/modalities/sequence) | [Transformer](/concepts/architectures/transformer), [State-space model](/concepts/architectures/state-space-model) |
| Image / video | [Image](/concepts/modalities/image), [Video](/concepts/modalities/video) | [CNN](/concepts/architectures/cnn), [Vision Transformer](/concepts/architectures/vision-transformer) |
| Graph / set | [Graph](/concepts/modalities/graph) | [GNN](/concepts/architectures/gnn), [Deep Sets](/concepts/architectures/deep-sets) |
| 3D / geometry | [3D structure](/concepts/modalities/3d-structure), [Geometry](/concepts/math/geometry) | [Geometric deep learning](/concepts/geometric-deep-learning) |
| Molecule / protein | [Computational Biology](/molecular-modeling) | [Molecular modeling](/concepts/molecular-modeling), [Protein modeling](/concepts/protein-modeling), [SBDD concepts](/concepts/sbdd) |
| Agent workflow | [Agents](/agents) | [Core](/agents/core), [Tools](/agents/tools), [Verification](/agents/verification), [Workflows](/agents/workflows) |

## 논문과 포스트를 읽을 때

새 AI 논문이나 글감을 넣을 때는 모델 이름보다 아래 항목을 먼저 고정합니다.

| 먼저 볼 것 | 확인할 내용 | Start |
| --- | --- | --- |
| Input object | text, image, graph, set, coordinate, molecule, protein, agent state 중 무엇인가 | [Modalities](/concepts/modalities), [Tasks](/concepts/tasks) |
| Task output | class, scalar, ranking, sequence, graph, coordinate, sample, action 중 무엇인가 | [Task specification](/concepts/tasks/task-specification), [Task output space](/concepts/tasks/task-output-space) |
| Representation | raw object가 token, graph, coordinate, embedding, conformer로 어떻게 바뀌는가 | [Representation contract](/concepts/modalities/representation-contract) |
| Architecture | 어떤 inductive bias, parameter sharing, complexity를 쓰는가 | [Architectures](/ai/architectures) |
| Learning signal | label, mask, contrast, preference, reward, denoising, velocity 중 무엇인가 | [Learning Methods](/ai/learning-methods) |
| Objective | loss, likelihood, score, reward, metric이 어떻게 정의되는가 | [Machine Learning](/ai/machine-learning), [Math](/math) |
| Evaluation claim | 어떤 split, metric, baseline, uncertainty로 주장을 검증하는가 | [Evaluation](/ai/evaluation) |
| System boundary | training, inference, serving, reproducibility, tool-use 문제가 있는가 | [Systems](/concepts/systems), [Infra](/infra), [Agents](/agents) |
| Intake protocol | 위 항목들을 한 번에 점검할 paper note인가 | [AI paper intake](/ai/paper-intake) |
| Claim pattern | architecture, learning, generation, evaluation, scaling, systems, agent 중 어떤 claim인가 | [AI paper claim patterns](/ai/paper-claim-patterns) |
| Cross-axis contract | computational biology 대상과 수식이 함께 걸린 AI claim인가 | [AI Computational Biology Math contract](/concepts/ai-computational-biology-math-contract) |

## Related

- [[concepts/index|Concepts]]
- [[math/index|Math]]
- [[molecular-modeling/index|Computational Biology]]
- [[infra/index|Infra]]
- [[agents/index|Agents]]
- [[ai/paper-intake|AI paper intake]]
- [[ai/paper-claim-patterns|AI paper claim patterns]]
- [[concepts/ai-computational-biology-math-contract|AI Computational Biology Math contract]]
- [[concepts/coverage-matrix|Coverage matrix]]
