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

- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/softmax|Softmax]]
- [[concepts/architectures/dropout|Dropout]]
- [[concepts/architectures/mlp|MLP]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder]]

## Spatial and Sequential Models

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/state-space-model|State-space models]]
- [[concepts/architectures/mamba|Mamba]]

## Graph and Sparse Models

- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-transformer|Graph Transformer]]
- [[concepts/architectures/mixture-of-experts|Mixture of Experts]]

## Reading Checklist

- What is the input object: vector, sequence, grid, graph, structure, or mixed modality?
- What mixes information: dense layers, convolution, recurrence, state update, message passing, or attention?
- What inductive bias is assumed: locality, order, permutation invariance, equivariance, sparsity, or routing?
- What fails first in practice: memory, length, graph size, data leakage, calibration, or decoding constraints?

## Related

- [[entities/index|Entities]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/learning/index|Learning methods]]
- [[concepts/generative-models/index|Generative models]]
- [[research/protein-modeling/index|Protein modeling]]
