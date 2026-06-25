---
title: Encoder-Only Transformer
tags:
  - architectures
  - transformer
  - representation-learning
---

# Encoder-Only Transformer

An encoder-only Transformer uses bidirectional self-attention to build contextual representations. It is commonly used for classification, retrieval, masked modeling, and representation learning.

The self-attention mask usually allows all non-padding tokens to attend to each other:

$$
M_{ij} =
\begin{cases}
0, & \text{if token } j \text{ is visible} \\
-\infty, & \text{if token } j \text{ is padding}
\end{cases}
$$

The output can be token-level states or a pooled representation:

$$
h_{1:T} = f_\theta(x_{1:T})
$$

$$
z = \operatorname{Readout}(h_{1:T})
$$

## Uses

- Masked language or residue modeling.
- Protein or molecule representation learning.
- Classification and retrieval.
- Feature extraction for downstream models.

## Checks

- Is the task representation learning rather than generation?
- How are masked tokens, padding, and special tokens handled?
- Is the readout CLS, mean pooling, attention pooling, or task-specific?

## Related

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
