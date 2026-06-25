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

## Core Notes

- [[concepts/architectures/index|Architecture index]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/softmax|Softmax]]
- [[concepts/architectures/mlp|MLP]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-transformer|Graph transformer]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/mamba|Mamba]]
- [[concepts/architectures/mixture-of-experts|Mixture of experts]]

## Questions

- 입력이 sequence, graph, set, grid, structure 중 무엇인가?
- 모델이 locality, order, permutation invariance, equivariance 중 무엇을 가정하는가?
- long-context scaling, sparse routing, geometric bias가 필요한가?
- architecture의 inductive bias가 task의 symmetry와 맞는가?

## Related

- [[ai/learning-methods|Learning methods]]
- [[ai/generative-models|Generative models]]
- [[bio-ai/index|Bio-AI]]
