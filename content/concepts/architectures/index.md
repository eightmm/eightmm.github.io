---
title: Architectures
tags:
  - architectures
  - machine-learning
---

# Architectures

Architecture notes describe model families used across molecular AI, protein modeling, agents, and implementation reading. Use these pages as quick checks for what a paper is mixing, what inductive bias it assumes, and where evaluation can go wrong.

An architecture defines the function class searched during training:

$$
f_\theta \in \mathcal{F}_{\text{arch}}
$$

The same training objective can behave very differently depending on whether $\mathcal{F}_{\text{arch}}$ encodes locality, order, graph structure, or symmetry.

## Route Map

| Question | Start | Then Check |
| --- | --- | --- |
| what input object is being modeled? | [Modalities](/concepts/modalities), [Architecture selection](/concepts/architectures/architecture-selection) | sequence, graph, grid, set, coordinate, or mixed input |
| what inductive bias is assumed? | [Inductive bias](/concepts/architectures/inductive-bias), [Parameter sharing](/concepts/architectures/parameter-sharing) | locality, order, permutation, symmetry, sparsity |
| what mixes information? | [Attention](/concepts/architectures/attention), [Convolution](/concepts/architectures/convolution), [GNN](/concepts/architectures/gnn), [State-space models](/concepts/architectures/state-space-model) | token, pixel, node, coordinate, or state mixing |
| what transforms channels? | [Linear layer](/concepts/architectures/linear-layer), [Feed-forward network](/concepts/architectures/feed-forward-network), [Activation function](/concepts/architectures/activation-function) | width, expansion, gating, parameter budget |
| what stabilizes depth? | [Residual connection](/concepts/architectures/residual-connection), [Normalization](/concepts/architectures/normalization), [Normalization placement](/concepts/architectures/normalization-placement), [Weight initialization](/concepts/architectures/weight-initialization) | gradient flow and scale |
| what maps internal states to outputs? | [Pooling and readout](/concepts/architectures/pooling-readout), [Softmax](/concepts/architectures/softmax) | task output space and metric |
| what is the cost bottleneck? | [Computational complexity](/concepts/architectures/computational-complexity) | length, graph size, resolution, experts, memory |

## Basic Blocks

| Group | Notes |
| --- | --- |
| Dense transforms | [Linear layer](/concepts/architectures/linear-layer), [Activation function](/concepts/architectures/activation-function), [Feed-forward network](/concepts/architectures/feed-forward-network), [Gating](/concepts/architectures/gating) |
| Training stability | [Weight initialization](/concepts/architectures/weight-initialization), [Normalization](/concepts/architectures/normalization), [Normalization placement](/concepts/architectures/normalization-placement), [Residual connection](/concepts/architectures/residual-connection), [Dropout](/concepts/architectures/dropout) |
| Input representation | [Tokenization](/concepts/architectures/tokenization), [Embedding](/concepts/architectures/embedding), [Positional encoding](/concepts/architectures/positional-encoding), [Graph construction](/concepts/architectures/graph-construction) |
| Readout | [Pooling and readout](/concepts/architectures/pooling-readout), [Softmax](/concepts/architectures/softmax) |

## Dense and Feed-Forward Models

- [[concepts/architectures/mlp|MLP]]
- [[concepts/architectures/autoencoder|Autoencoder]]

## Grid, Image, and Voxel Models

- [[concepts/modalities/image|Image]]
- [[concepts/modalities/video|Video]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/residual-network|Residual network]]
- [[concepts/architectures/u-net|U-Net]]
- [[concepts/architectures/vision-transformer|Vision Transformer]]

## Sequence Models

- [[concepts/modalities/text|Text]]
- [[concepts/modalities/audio|Audio]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/lstm|LSTM]]
- [[concepts/architectures/gru|GRU]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/encoder-only-transformer|Encoder-only Transformer]]
- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/architectures/state-space-model|State-space models]]
- [[concepts/architectures/mamba|Mamba]] as a selective state-space model

## Attention and Encoder-Decoder Patterns

- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder]]
- [[concepts/architectures/perceiver|Perceiver]]

## Set, Graph, and Sparse Models

- [[concepts/architectures/deep-sets|Deep Sets]]
- [[concepts/architectures/set-transformer|Set Transformer]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-transformer|Graph Transformer]]
- [[concepts/architectures/mixture-of-experts|Mixture of Experts]]

## Reading Checklist

- What is the input object: vector, sequence, grid, graph, structure, or mixed modality?
- What [[concepts/architectures/inductive-bias|inductive bias]] is being assumed?
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
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/learning/index|Learning methods]]
- [[concepts/generative-models/index|Generative models]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[molecular-modeling/protein-modeling|Protein modeling]]
