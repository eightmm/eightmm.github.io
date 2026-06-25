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

## Basic Blocks

- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/architectures/parameter-sharing|Parameter sharing]]
- [[concepts/architectures/architecture-selection|Architecture selection]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/residual-network|Residual network]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/softmax|Softmax]]
- [[concepts/architectures/dropout|Dropout]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]

## Dense and Feed-Forward Models

- [[concepts/architectures/mlp|MLP]]
- [[concepts/architectures/autoencoder|Autoencoder]]

## Grid, Image, and Voxel Models

- [[concepts/modalities/image|Image]]
- [[concepts/modalities/video|Video]]
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
- [[research/protein-modeling/index|Protein modeling]]
