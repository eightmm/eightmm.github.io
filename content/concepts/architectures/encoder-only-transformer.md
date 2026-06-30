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

The bidirectional attention update is:

$$
\operatorname{Attn}(H)
=
\operatorname{softmax}
\left(
\frac{HW_Q (HW_K)^\top}{\sqrt{d_k}} + M_{\mathrm{pad}}
\right)HW_V
$$

where $M_{\mathrm{pad}}$ masks padding but does not impose causal order. This makes every visible token representation depend on both left and right context:

$$
h_t = f_\theta(x_t, x_{\setminus t})
$$

Some encoder-only variants change how position enters this attention equation. [[papers/architectures/deberta|DeBERTa]], for example, separates content vectors from relative position vectors when computing attention logits:

$$
\operatorname{score}(i,j)
=
\text{content-content}
+ \text{content-position}
+ \text{position-content}.
$$

This keeps the encoder-only contract while changing the position/content factorization inside attention.

## Objective Fit

Encoder-only models are good when the downstream task needs representations rather than left-to-right generation. Common objectives include:

- Masked modeling, where hidden tokens are reconstructed from context.
- Contrastive or retrieval objectives, where sequence-level vectors are compared.
- Token classification, where every token/residue/node gets a label.
- Pair or cross-input scoring, where pooled representations feed a classifier or ranker.

For protein and molecule use, readout details matter: pooling over padding, special tokens, or truncated residues can silently corrupt the representation.

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
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[papers/architectures/bert|BERT]]
- [[papers/architectures/deberta|DeBERTa]]
