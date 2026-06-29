---
title: Architectures
tags:
  - ai
  - architectures
---

# Architectures

모델 아키텍처는 데이터가 어떤 구조를 갖는지에 대한 가정입니다. 같은 objective를 쓰더라도 sequence, graph, image, 3D structure를 어떻게 표현하느냐에 따라 적합한 architecture가 달라집니다.

형식적으로는 architecture가 가능한 함수 공간 $\mathcal{F}$를 정합니다.

$$
f_\theta \in \mathcal{F}_{\text{architecture}}
$$

예를 들어 CNN은 locality와 translation equivariance를 강하게 가정하고, GNN은 graph connectivity와 permutation invariance/equivariance를 가정합니다.

## 고르는 기준

아키텍처를 볼 때는 "유명한 모델인가"보다 "어떤 입력 구조와 대칭성을 가정하는가"를 먼저 봐야 합니다.

| Criterion | Start | Ask |
| --- | --- | --- |
| Inductive bias | [Inductive bias](/concepts/architectures/inductive-bias) | architecture가 어떤 함수 공간을 선호하는가? |
| Parameter sharing | [Parameter sharing](/concepts/architectures/parameter-sharing) | 같은 parameter를 어디에 반복 적용하는가? |
| Architecture choice | [Architecture selection](/concepts/architectures/architecture-selection) | modality, task, data, compute에 맞는 family인가? |
| Architecture vs objective | [Architecture-objective fit](/concepts/architectures/architecture-objective-fit) | 성능 차이가 구조 때문인지 학습 신호 때문인지 분리되는가? |
| Scaling cost | [Computational complexity](/concepts/architectures/computational-complexity) | sequence length, graph size, image resolution에 따라 비용이 어떻게 느는가? |

## Architecture Reading Order

아키텍처 노트는 아래 순서로 읽으면 산발적인 모델명이 하나의 체계로 들어옵니다.

1. 입력 단위: feature, token, pixel, node, residue, atom, coordinate 중 무엇인가?
2. 섞는 축: channel, token, node, edge, coordinate, modality 중 무엇을 mix하는가?
3. 공유 규칙: convolution, recurrence, attention, message passing, equivariant update 중 무엇인가?
4. 안정화 장치: normalization, residual, gating, initialization이 어디에 놓이는가?
5. 출력 readout: class, scalar, sequence, graph, coordinate, distribution 중 무엇을 내는가?
6. 비용: sequence length, graph size, resolution, coordinate updates에 대해 어떻게 scaling되는가?

## Input Structure Map

| Input Structure | Useful Bias | Typical Architecture | Watch |
| --- | --- | --- | --- |
| Fixed feature vector | feature mixing | MLP, linear model, tree model | feature scale, missing values, leakage |
| Grid / image / voxel | local translation sharing | CNN, U-Net, ViT | resolution, receptive field, augmentation |
| Sequence | order and long-range context | RNN/LSTM/GRU, Transformer, SSM/Mamba | length scaling, positional encoding, truncation |
| Set | permutation invariance | Deep Sets, Set Transformer | pooling/readout and element identity |
| Graph | neighborhood message passing or graph-biased attention | GNN, Graph Transformer | graph construction, edge attributes, over-smoothing |
| 3D coordinates | invariance/equivariance | equivariant GNN, tensor-field model | coordinate frame, units, chirality, leakage, update stability |
| Multimodal input | cross-modal alignment | encoder-decoder, cross-attention, Perceiver | missing modality, modality leakage, fusion timing |

## Building Block Routes

딥러닝 아키텍처를 읽을 때 먼저 봐야 하는 building block입니다.

| Block | Start | Role |
| --- | --- | --- |
| Affine transform | [Linear layer](/concepts/architectures/linear-layer), [Weight initialization](/concepts/architectures/weight-initialization) | feature dimension mixing and stable starting scale |
| Nonlinearity | [Activation function](/concepts/architectures/activation-function), [Feed-forward network](/concepts/architectures/feed-forward-network) | local nonlinear transformation and channel mixing |
| Stabilization | [Normalization](/concepts/architectures/normalization), [Normalization placement](/concepts/architectures/normalization-placement), [Residual connection](/concepts/architectures/residual-connection) | gradient flow and deep model stability |
| Representation interface | [Tokenization](/concepts/architectures/tokenization), [Embedding](/concepts/architectures/embedding), [Positional encoding](/concepts/architectures/positional-encoding) | mapping raw objects into model states |
| Probability and regularization | [Softmax](/concepts/architectures/softmax), [Dropout](/concepts/architectures/dropout) | normalized scores and train-time noise |
| Routing and readout | [Gating](/concepts/architectures/gating), [Pooling and readout](/concepts/architectures/pooling-readout) | conditional computation and output aggregation |

## Architecture Families

| Family | Start | Best Fit | Watch |
| --- | --- | --- | --- |
| Dense / feed-forward | [MLP](/concepts/architectures/mlp), [Autoencoder](/concepts/architectures/autoencoder) | fixed vectors, tabular features, bottleneck representations | feature scaling and lost structure |
| Grid / image / voxel | [Image](/concepts/modalities/image), [Video](/concepts/modalities/video), [Convolution](/concepts/architectures/convolution), [CNN](/concepts/architectures/cnn), [U-Net](/concepts/architectures/u-net), [ViT](/concepts/architectures/vision-transformer) | locality, spatial sharing, dense prediction | receptive field, resolution, augmentation |
| Sequence recurrence | [RNN](/concepts/architectures/rnn), [LSTM](/concepts/architectures/lstm), [GRU](/concepts/architectures/gru) | ordered streams and compact state | long-range memory and parallelism limits |
| Attention / Transformer | [Attention](/concepts/architectures/attention), [Transformer](/concepts/architectures/transformer), [Encoder-only](/concepts/architectures/encoder-only-transformer), [Decoder-only](/concepts/architectures/decoder-only-transformer), [Encoder-decoder](/concepts/architectures/encoder-decoder) | long-range token interaction and conditional generation | quadratic context cost and position handling |
| State-space sequence models | [State-space model](/concepts/architectures/state-space-model), [Mamba](/concepts/architectures/mamba) | long sequences with recurrent state and parallel training | task fit, selective state design, comparison to attention |
| Cross-modal patterns | [Multimodal learning](/concepts/modalities/multimodal-learning), [Cross-attention](/concepts/architectures/cross-attention), [Perceiver](/concepts/architectures/perceiver) | fusing different input types or context sources | modality leakage and fusion timing |
| Set and graph | [Deep Sets](/concepts/architectures/deep-sets), [Set Transformer](/concepts/architectures/set-transformer), [Graph construction](/concepts/architectures/graph-construction), [GNN](/concepts/architectures/gnn), [Graph Transformer](/concepts/architectures/graph-transformer) | unordered elements, molecules, residues, relations | edge definition, readout, over-smoothing |
| Geometry-aware | [Geometry](/concepts/math/geometry), [Symmetry group](/concepts/math/symmetry-group), [Geometric deep learning](/concepts/geometric-deep-learning), [Equivariance](/concepts/geometric-deep-learning/equivariance), [Equivariant GNN](/concepts/geometric-deep-learning/equivariant-gnn) | coordinates, forces, poses, molecular and protein structures | coordinate frame, invariance/equivariance split, chirality |
| Sparse / routed | [Mixture of experts](/concepts/architectures/mixture-of-experts), [Gating](/concepts/architectures/gating) | conditional compute and large-capacity models | routing collapse, load balance, serving cost |

Mamba는 별도 대분류가 아니라 [state-space model](/concepts/architectures/state-space-model) 계열의 selective SSM으로 읽습니다.

## Architecture Claim Checklist

| Claim | Check |
| --- | --- |
| Better inductive bias | Does the architecture match the input symmetry or locality? |
| Better scaling | What is the compute/memory complexity in sequence length, nodes, or pixels? |
| Better representation | Is the readout or pooling rule aligned with the task output? |
| Better generation | Does the architecture support the sampling path and conditioning interface? |
| Better structure modeling | Are invariant and equivariant quantities handled separately? |
| Better graph modeling | Is graph construction part of the method or held fixed across baselines? |
| Better coordinate generation | Are coordinate update step count, constraints, and geometry validity reported? |

## 읽을 때 볼 질문

- 입력이 sequence, graph, set, grid, structure 중 무엇인가?
- 모델이 locality, order, permutation invariance, equivariance 중 무엇을 가정하는가?
- Linear layer는 channel을 섞는가, attention/convolution/message passing은 token이나 neighborhood를 섞는가?
- Activation, normalization, residual path가 gradient flow를 안정화하는가?
- Weight initialization과 normalization placement가 깊은 모델의 안정성에 맞는가?
- long-context scaling, sparse routing, geometric bias가 필요한가?
- architecture의 inductive bias가 task의 symmetry와 맞는가?

## Related

- [[ai/learning-methods|Learning methods]]
- [[ai/generative-models|Generative models]]
- [[concepts/modalities/index|Modalities]]
- [[molecular-modeling/index|Computational Biology]]
- [[concepts/protein-modeling/index|Protein modeling concepts]]
