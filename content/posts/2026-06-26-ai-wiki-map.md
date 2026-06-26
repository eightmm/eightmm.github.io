---
title: AI Wiki를 어떤 축으로 나눠 볼 것인가
date: 2026-06-26
tags:
  - posts
  - ai
  - wiki
---

# AI Wiki를 어떤 축으로 나눠 볼 것인가

AI를 정리할 때 가장 쉽게 빠지는 방식은 모델 이름을 나열하는 것입니다. Transformer, GNN, diffusion, Mamba, MoE처럼 이름을 따라가면 최신 흐름은 빨리 볼 수 있지만, 왜 그 구조가 필요한지와 어떤 문제에 맞는지는 흐려집니다.

이 wiki에서는 모델 이름보다 먼저 다음 질문을 봅니다.

$$
\text{AI problem}
=
(\text{entity}, \text{modality}, \text{task}, \text{architecture}, \text{learning}, \text{evaluation}, \text{system})
$$

이렇게 나누면 text, image, molecule, protein, graph, 3D structure, agent workflow가 한곳에 섞여도 길을 잃지 않습니다.

## Entity

Entity는 모델이 다루는 대상입니다. Molecular modeling에서는 [[entities/protein|Protein]], [[entities/ligand|Ligand]], [[entities/molecule|Molecule]], [[entities/pocket|Pocket]], [[entities/protein-ligand-complex|Protein-ligand complex]], [[entities/assay|Assay]], [[entities/dataset|Dataset]]이 먼저 정리되어야 합니다.

같은 graph model을 쓰더라도 molecule graph와 protein contact graph는 의미가 다릅니다. 같은 label이라도 molecule-only property인지, target-conditioned activity인지, assay-conditioned measurement인지에 따라 split과 metric이 달라집니다. 그래서 [[entities/entity-relation-map|Entity relation map]]과 [[entities/target-assay-label|Target-assay-label contract]]는 Molecular Modeling 글의 출발점입니다.

## Modality

Modality는 입력과 출력 신호의 형태입니다. [[concepts/modalities/text|Text]], [[concepts/modalities/image|Image]], [[concepts/modalities/video|Video]], [[concepts/modalities/sequence|Sequence]], [[concepts/modalities/graph|Graph]], [[concepts/modalities/3d-structure|3D structure]]는 preprocessing, tokenization, missing information, leakage risk가 다릅니다.

이 블로그에서는 entity와 modality를 분리해서 봅니다. Protein은 sequence로 볼 수도 있고, structure로 볼 수도 있고, graph로 볼 수도 있습니다. Molecule도 SMILES sequence, molecular graph, fingerprint, conformer ensemble로 표현될 수 있습니다. 이 연결은 [[concepts/modalities/modality-representation|Modality representation]]과 [[concepts/modalities/modality-task-map|Modality-task map]]에서 관리합니다.

## Task

Task는 모델이 무엇을 출력해야 하는지입니다. Classification, regression, retrieval, ranking, generation, segmentation, coordinate prediction은 모두 다른 output space를 갖습니다.

$$
f_\theta: \mathcal{X} \rightarrow \mathcal{Y}
$$

여기서 $\mathcal{Y}$가 class인지, scalar인지, sequence인지, graph인지, coordinate인지에 따라 loss와 metric이 바뀝니다. 그래서 [[concepts/tasks/task-specification|Task specification]]과 [[concepts/tasks/task-output-space|Task output space]]를 architecture보다 먼저 봅니다.

## Architecture

Architecture는 입력 구조와 inductive bias를 정합니다. [[concepts/architectures/cnn|CNN]]은 local grid structure에 강하고, [[concepts/architectures/rnn|RNN]]과 [[concepts/architectures/state-space-model|State-space model]]은 sequence 처리 방식이 다르며, [[concepts/architectures/transformer|Transformer]]는 attention으로 token 간 상호작용을 직접 모델링합니다.

Graph와 structure가 중요해지면 [[concepts/architectures/gnn|GNN]], [[concepts/architectures/graph-transformer|Graph transformer]], [[concepts/geometric-deep-learning/equivariance|Equivariance]], [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]] 같은 개념이 필요합니다. 이때 Mamba는 독립 대분류라기보다 [[concepts/architectures/state-space-model|state-space model]] 계열의 한 구현으로 봅니다.

## Learning

Learning method는 어떤 신호로 표현을 학습하는지입니다. [[concepts/learning/supervised-learning|Supervised learning]], [[concepts/learning/self-supervised-learning|Self-supervised learning]], [[concepts/learning/contrastive-learning|Contrastive learning]], [[concepts/learning/jepa|JEPA]], [[concepts/learning/fine-tuning|Fine-tuning]], [[concepts/learning/preference-optimization|Preference optimization]]은 서로 다른 data requirement와 failure mode를 갖습니다.

Representation learning을 볼 때는 pretraining objective와 downstream evaluation을 분리해야 합니다. 좋은 pretraining loss가 곧 좋은 downstream performance를 보장하지 않기 때문에 [[concepts/learning/representation-evaluation|Representation evaluation]]과 [[concepts/learning/linear-probing|Linear probing]]을 함께 봅니다.

## Generation

생성 모델은 data distribution을 근사하거나 조건에 맞는 sample을 만드는 계열입니다.

$$
x \sim p_\theta(x \mid c)
$$

[[concepts/generative-models/autoregressive-model|Autoregressive model]], [[concepts/generative-models/diffusion-model|Diffusion model]], [[concepts/generative-models/flow-matching|Flow matching]], [[concepts/generative-models/normalizing-flow|Normalizing flow]], [[concepts/generative-models/vae|VAE]], [[concepts/generative-models/gan|GAN]]은 sampling 과정과 학습 target이 다릅니다.

Molecular modeling에서는 이 축이 [[concepts/generative-models/molecular-generation|Molecular generation]]과 [[concepts/generative-models/protein-design|Protein design]]으로 이어집니다. 여기서는 validity, novelty, diversity, controllability, task utility를 분리해서 평가해야 합니다.

## Evaluation

Evaluation은 모델 비교를 지식으로 바꾸는 기준입니다. metric만 보면 부족하고, split, benchmark, baseline, uncertainty, failure mode를 같이 봐야 합니다.

$$
\hat{R}(f)
=
\frac{1}{n}\sum_{i=1}^{n}\mathcal{L}(f(x_i), y_i)
$$

이 추정값이 의미 있으려면 test set이 원하는 generalization setting을 대표해야 합니다. 그래서 [[concepts/evaluation/evaluation-protocol|Evaluation protocol]], [[concepts/evaluation/metric-selection|Metric selection]], [[concepts/evaluation/leakage|Leakage]], [[concepts/evaluation/ood-generalization|OOD generalization]], [[concepts/evaluation/calibration|Calibration]]을 계속 연결합니다.

## System

모델은 코드와 논문 속에서만 존재하지 않습니다. 실제로는 data loading, GPU memory, distributed training, checkpoint, inference serving, reproducibility가 붙습니다.

이 축은 [[infra/index|Infra]]와 [[concepts/systems/index|AI systems]]에서 관리합니다. 예를 들어 training run은 model architecture만의 문제가 아니라 [[infra/gpu/index|GPU]], [[infra/hpc/index|HPC]], [[infra/reproducibility/index|Reproducibility]], [[concepts/systems/run-artifact|Run artifact]]의 문제이기도 합니다.

## 읽는 순서

이 wiki를 처음 읽는다면 아래 순서가 좋습니다.

1. [[math/index|Math]]에서 공통 수식을 잡습니다.
2. [[ai/machine-learning|Machine learning]]에서 loss, optimization, generalization을 봅니다.
3. [[ai/architectures|Architectures]]에서 모델 family를 입력 구조별로 나눕니다.
4. [[ai/learning-methods|Learning methods]]에서 supervision과 representation learning을 봅니다.
5. [[ai/generative-models|Generative models]]에서 sampling과 distribution modeling을 봅니다.
6. [[ai/evaluation|Evaluation]]에서 split, metric, uncertainty, failure mode를 점검합니다.
7. [[molecular-modeling/index|Computational Biology]]에서 protein, molecule, ligand, structure-based modeling로 좁힙니다.
8. [[infra/index|Infra]]와 [[agents/index|Agents]]에서 실제 workflow와 운영을 연결합니다.

목표는 모든 글을 하나의 taxonomy에 억지로 넣는 것이 아닙니다. 글마다 가장 강한 축을 정하고, 나머지 축은 wikilink로 연결하는 것입니다. 그렇게 하면 블로그는 한글 narrative가 되고, wiki는 영어 canonical note가 됩니다.
