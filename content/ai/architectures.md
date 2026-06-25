---
title: Architectures
tags:
  - ai
  - architectures
---

# Architectures

모델 아키텍처는 데이터가 어떤 구조를 갖는지에 대한 가정입니다. 같은 objective를 쓰더라도 sequence, graph, image, 3D structure를 어떻게 표현하느냐에 따라 적합한 architecture가 달라집니다.

이 페이지는 한글 안내 페이지입니다. 링크된 `concepts/architectures/*` 문서는 영어 canonical wiki note로 유지합니다.

형식적으로는 architecture가 가능한 함수 공간 $\mathcal{F}$를 정합니다.

$$
f_\theta \in \mathcal{F}_{\text{architecture}}
$$

예를 들어 CNN은 locality와 translation equivariance를 강하게 가정하고, GNN은 graph connectivity와 permutation invariance/equivariance를 가정합니다.

## 기본 구성요소

딥러닝 아키텍처를 읽을 때 먼저 봐야 하는 building block입니다.

- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/softmax|Softmax]]
- [[concepts/architectures/dropout|Dropout]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]

## Dense and Feed-Forward

고정 길이 vector나 이미 만들어진 feature를 처리하는 가장 기본적인 계열입니다.

- [[concepts/architectures/mlp|MLP]]
- [[concepts/architectures/autoencoder|Autoencoder]]

## Grid, Image, and Voxel Models

locality와 weight sharing이 중요한 입력에 적합합니다. 이미지뿐 아니라 contact map, voxelized structure, spatial grid에도 연결됩니다.

- [[concepts/modalities/image|Image]]
- [[concepts/modalities/video|Video]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/residual-network|Residual network]]
- [[concepts/architectures/u-net|U-Net]]
- [[concepts/architectures/vision-transformer|Vision Transformer]]

## Sequence Models

token, residue, text, time series처럼 순서가 있는 입력을 다룹니다.

- [[concepts/modalities/text|Text]]
- [[concepts/modalities/audio|Audio]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/lstm|LSTM]]
- [[concepts/architectures/gru|GRU]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/encoder-only-transformer|Encoder-only Transformer]]
- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/mamba|Mamba]]

Mamba는 별도 대분류가 아니라 [[concepts/architectures/state-space-model|state-space model]] 계열의 selective SSM으로 봅니다.

## Attention and Encoder-Decoder Patterns

sequence, graph, multimodal input을 섞는 공통 패턴입니다.

- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder]]
- [[concepts/architectures/perceiver|Perceiver]]

## Set and Graph Models

순서가 없거나 관계 구조가 중요한 객체를 다룹니다.

- [[concepts/architectures/deep-sets|Deep Sets]]
- [[concepts/architectures/set-transformer|Set Transformer]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-transformer|Graph transformer]]

## Geometry-Aware Models

3D coordinate, molecular structure, protein complex처럼 회전과 이동에 대한 대칭성이 중요한 입력을 다룹니다. 순수한 geometry 개념은 [[concepts/math/geometry|Geometry]]와 [[concepts/math/symmetry-group|Symmetry group]]에 두고, AI 쪽에서는 그것을 모델 구조로 구현하는 [[concepts/geometric-deep-learning/index|Geometric deep learning]]을 봅니다.

- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/geometric-deep-learning/geometric-architecture|Geometric architecture]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]

## Sparse and Routed Models

모든 parameter를 항상 쓰지 않고 routing이나 sparsity를 활용합니다.

- [[concepts/architectures/mixture-of-experts|Mixture of experts]]

## 읽을 때 볼 질문

- 입력이 sequence, graph, set, grid, structure 중 무엇인가?
- 모델이 locality, order, permutation invariance, equivariance 중 무엇을 가정하는가?
- long-context scaling, sparse routing, geometric bias가 필요한가?
- architecture의 inductive bias가 task의 symmetry와 맞는가?

## Related

- [[ai/learning-methods|Learning methods]]
- [[ai/generative-models|Generative models]]
- [[concepts/modalities/index|Modalities]]
- [[bio-ai/index|Bio-AI]]
- [[concepts/protein-modeling/index|Protein modeling concepts]]
