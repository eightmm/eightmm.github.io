---
title: Architectures
tags:
  - architectures
  - machine-learning
---

# Architectures

Architecture note는 molecular AI, protein modeling, agent, 구현 읽기에서 쓰는 model family를 설명합니다. 논문이 무엇을 섞고 있는지, 어떤 inductive bias를 가정하는지, evaluation이 어디서 흔들릴 수 있는지 빠르게 확인하는 용도입니다.

Architecture는 training 중 탐색하는 function class를 정의합니다.

$$
f_\theta \in \mathcal{F}_{\text{arch}}
$$

같은 training objective라도 $\mathcal{F}_{\text{arch}}$가 locality, order, graph structure, symmetry 중 무엇을 encode하는지에 따라 전혀 다르게 동작할 수 있습니다.

## Architecture Contract

Architecture note는 모델명을 외우는 것이 아니라, 입력 구조와 정보 흐름을 계약으로 고정합니다.

$$
\mathcal{A}
=
(\mathcal{X},\ H,\ M,\ S,\ R,\ O,\ C)
$$

| Part | Meaning | Typical question |
| --- | --- | --- |
| $\mathcal{X}$ | input object | vector, sequence, grid, graph, set, coordinate, multimodal input? |
| $H$ | hidden state shape | token states, node states, feature maps, coordinates, latent vectors? |
| $M$ | mixing rule | dense layer, convolution, recurrence, attention, message passing, state update? |
| $S$ | sharing or symmetry rule | translation sharing, permutation invariance, equivariance, causal order? |
| $R$ | readout | class token, pooling, graph readout, coordinate head, distribution head? |
| $O$ | objective compatibility | supervised loss, generative sampler, SSL target, ranking objective? |
| $C$ | compute and memory cost | sequence length, graph size, resolution, state size, expert count? |

이 계약을 먼저 쓰면 “Transformer vs Mamba”, “GNN vs Graph Transformer”, “CNN vs ViT” 같은 비교가 모델명 싸움이 아니라 assumption 비교가 됩니다.

## Route Map

| Question | Start | Then Check |
| --- | --- | --- |
| what input object is being modeled? | [Modalities](/concepts/modalities), [Architecture selection](/concepts/architectures/architecture-selection) | sequence, graph, grid, set, coordinate, or mixed input |
| is the claim architecture or objective driven? | [Architecture-objective fit](/concepts/architectures/architecture-objective-fit) | data, objective, compute, and evaluation controls |
| what inductive bias is assumed? | [Inductive bias](/concepts/architectures/inductive-bias), [Parameter sharing](/concepts/architectures/parameter-sharing) | locality, order, permutation, symmetry, sparsity |
| what mixes information? | [Attention](/concepts/architectures/attention), [Convolution](/concepts/architectures/convolution), [GNN](/concepts/architectures/gnn), [State-space models](/concepts/architectures/state-space-model) | token, pixel, node, coordinate, or state mixing |
| what transforms channels? | [Linear layer](/concepts/architectures/linear-layer), [Feed-forward network](/concepts/architectures/feed-forward-network), [Activation function](/concepts/architectures/activation-function) | width, expansion, gating, parameter budget |
| what stabilizes depth? | [Residual connection](/concepts/architectures/residual-connection), [Normalization](/concepts/architectures/normalization), [Normalization placement](/concepts/architectures/normalization-placement), [Weight initialization](/concepts/architectures/weight-initialization) | gradient flow and scale |
| what maps internal states to outputs? | [Pooling and readout](/concepts/architectures/pooling-readout), [Softmax](/concepts/architectures/softmax) | task output space and metric |
| what is the cost bottleneck? | [Computational complexity](/concepts/architectures/computational-complexity) | length, graph size, resolution, experts, memory |

## Information Mixing Axes

Architecture family는 어떤 axis를 섞는지로 읽으면 정리가 쉽습니다.

| Mixing axis | Common block | Use when |
| --- | --- | --- |
| Feature/channel | [[concepts/architectures/linear-layer|Linear layer]], [[concepts/architectures/feed-forward-network|Feed-forward network]], [[concepts/architectures/mlp|MLP]] | each sample is already a vector or token state |
| Local spatial neighborhood | [[concepts/architectures/convolution|Convolution]], [[concepts/architectures/cnn|CNN]], [[concepts/architectures/u-net|U-Net]] | grid, image, voxel, dense spatial map |
| Ordered time or sequence state | [[concepts/architectures/rnn|RNN]], [[concepts/architectures/lstm|LSTM]], [[concepts/architectures/gru|GRU]], [[concepts/architectures/state-space-model|State-space model]] | recurrence, streaming, long sequence state |
| Global token interaction | [[concepts/architectures/attention|Attention]], [[concepts/architectures/transformer|Transformer]] | long-range interaction and flexible context |
| Cross-context interaction | [[concepts/architectures/cross-attention|Cross-attention]], [[concepts/architectures/encoder-decoder|Encoder-decoder]], [[concepts/architectures/perceiver|Perceiver]] | conditioning, retrieval, multimodal fusion |
| Graph neighborhood | [[concepts/architectures/gnn|GNN]], [[concepts/architectures/graph-transformer|Graph Transformer]] | molecular graphs, relation graphs, residue graphs |
| Set aggregation | [[concepts/architectures/deep-sets|Deep Sets]], [[concepts/architectures/set-transformer|Set Transformer]], [[concepts/architectures/pooling-readout|Pooling and readout]] | unordered elements and permutation-invariant output |
| Coordinate update | [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]], [[concepts/geometric-deep-learning/index|Geometric deep learning]] | 3D structure, force, pose, coordinate generation |
| Conditional compute | [[concepts/architectures/mixture-of-experts|Mixture of Experts]], [[concepts/architectures/gating|Gating]] | scaling capacity under sparse activation |

## Basic Blocks

| Group | Notes |
| --- | --- |
| Dense transforms | [Linear layer](/concepts/architectures/linear-layer), [Activation function](/concepts/architectures/activation-function), [Feed-forward network](/concepts/architectures/feed-forward-network), [Gating](/concepts/architectures/gating) |
| Training stability | [Weight initialization](/concepts/architectures/weight-initialization), [Normalization](/concepts/architectures/normalization), [Normalization placement](/concepts/architectures/normalization-placement), [Residual connection](/concepts/architectures/residual-connection), [Dropout](/concepts/architectures/dropout) |
| Input representation | [Tokenization](/concepts/architectures/tokenization), [Embedding](/concepts/architectures/embedding), [Positional encoding](/concepts/architectures/positional-encoding), [Graph construction](/concepts/architectures/graph-construction) |
| Readout | [Pooling and readout](/concepts/architectures/pooling-readout), [Softmax](/concepts/architectures/softmax) |

## Model Families

| Family | Use For | Start |
| --- | --- | --- |
| Dense / feed-forward | fixed vectors, tabular features, learned embeddings, bottlenecks | [MLP](/concepts/architectures/mlp), [Autoencoder](/concepts/architectures/autoencoder) |
| Grid / image / voxel | local spatial structure, image/video, contact maps, spatial grids | [Convolution](/concepts/architectures/convolution), [CNN](/concepts/architectures/cnn), [Residual network](/concepts/architectures/residual-network), [U-Net](/concepts/architectures/u-net), [Vision Transformer](/concepts/architectures/vision-transformer) |
| Sequence | text, residues, time series, ordered tokens | [RNN](/concepts/architectures/rnn), [LSTM](/concepts/architectures/lstm), [GRU](/concepts/architectures/gru), [Transformer](/concepts/architectures/transformer), [State-space models](/concepts/architectures/state-space-model), [Mamba](/concepts/architectures/mamba) |
| Attention / encoder-decoder | long-range mixing, conditioning, multimodal fusion | [Attention](/concepts/architectures/attention), [Cross-attention](/concepts/architectures/cross-attention), [Encoder-decoder](/concepts/architectures/encoder-decoder), [Perceiver](/concepts/architectures/perceiver) |
| Set / graph | unordered sets, molecular graphs, residue graphs, relational objects | [Deep Sets](/concepts/architectures/deep-sets), [Set Transformer](/concepts/architectures/set-transformer), [Graph construction](/concepts/architectures/graph-construction), [GNN](/concepts/architectures/gnn), [Weisfeiler-Lehman Test](/concepts/architectures/wl-test), [Graph Transformer](/concepts/architectures/graph-transformer) |
| Geometry-aware | coordinates, forces, poses, protein and molecule structures | [Geometric deep learning](/concepts/geometric-deep-learning), [Equivariant GNN](/concepts/geometric-deep-learning/equivariant-gnn), [Tensor Field Network](/concepts/geometric-deep-learning/tensor-field-network) |
| Sparse / routed | conditional compute, expert routing, scaling under budget | [Mixture of Experts](/concepts/architectures/mixture-of-experts) |

Mamba는 여기서 별도의 top-level family로 두지 않습니다. Selective state-space model이므로, Mamba 내부 구조 자체를 다루는 노트가 아니라면 [State-space models](/concepts/architectures/state-space-model) 경로로 둡니다.

## Comparison Controls

Architecture claim을 비교할 때는 architecture만 바뀌었는지 확인해야 합니다.

| Controlled factor | Why |
| --- | --- |
| data and split | stronger architecture claim needs the same train/test boundary |
| objective and target | different loss can explain the improvement |
| parameter count and compute | larger model can look like better inductive bias |
| context length or graph size | scaling advantage depends on input size |
| preprocessing and representation | better input encoding can dominate architecture |
| training budget and optimizer | undertrained baseline weakens comparison |
| evaluation metric and selection rule | architecture claim can be selection artifact |

The minimal comparison question is:

$$
\Delta \text{metric}
\stackrel{?}{=}
\Delta \text{architecture}
\quad
\text{given fixed data, objective, budget, and evaluation}
$$

## Reading Checklist

- What is the input object: vector, sequence, grid, graph, structure, or mixed modality?
- Where is [[concepts/architectures/parameter-sharing|parameter sharing]] used, and what symmetry does it encode?
- What mixes information: dense layers, convolution, recurrence, state update, message passing, or attention?
- Which blocks mix channels, tokens, spatial neighborhoods, graph neighborhoods, or coordinates?
- What inductive bias is assumed: locality, order, permutation invariance, equivariance, sparsity, or routing?
- What is the dominant [[concepts/architectures/computational-complexity|computational complexity]] term?
- How do activations, residual paths, normalization, and initialization affect gradient flow?
- What fails first in practice: memory, length, graph size, data leakage, calibration, or decoding constraints?

## Related

- [[entities/index|Entities]]
- [[concepts/modalities/index|Modalities]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/architectures/architecture-objective-fit|Architecture-objective fit]]
- [[concepts/architectures/wl-test|Weisfeiler-Lehman Test]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/learning/index|Learning methods]]
- [[concepts/generative-models/index|Generative models]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[molecular-modeling/protein-modeling|Protein modeling]]
