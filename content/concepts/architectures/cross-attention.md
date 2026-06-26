---
title: Cross-Attention
tags:
  - architectures
  - attention
  - multimodal
---

# Cross-Attention

Cross-attention mixes information from one representation into another. It is used when queries come from one sequence or object, while keys and values come from a different source.

The basic form is:

$$
\operatorname{CrossAttn}(X, C)
= \operatorname{softmax}\left(\frac{(XW_Q)(CW_K)^\top}{\sqrt{d_k}}\right)CW_V
$$

where $X$ provides queries and $C$ provides conditioning context. $C$ can be encoded text, image patches, retrieved documents, graph nodes, protein residues, ligand atoms, or tool observations.

For multi-head cross-attention:

$$
\operatorname{head}_m
=
\operatorname{softmax}
\left(
\frac{Q_mK_m^\top}{\sqrt{d_h}} + M
\right)V_m
$$

where $M$ is an optional mask over invalid or unavailable context entries.

## When It Appears

- Encoder-decoder Transformers, where decoder tokens attend to encoder states.
- Image or video generation conditioned on text.
- Multimodal models that align image, audio, video, and text features.
- Structure-based models where ligand representations attend to pocket or protein representations.
- Agent systems where a policy attends to retrieved context, memory, or tool outputs.

## Why It Matters

- Cross-attention preserves token-level access to the conditioning source.
- It can be more expressive than compressing the source into one pooled vector.
- Its memory cost scales with query length times context length.
- It can hide leakage if conditioning context contains labels or future information.

## Interface Contract

| Field | Question |
| --- | --- |
| Query object | Which tokens request information? |
| Context object | Which tokens provide keys and values? |
| Direction | Is conditioning one-way or bidirectional? |
| Mask | Which context entries are unavailable or invalid? |
| Positional relation | Are query-context distances, order, or geometry encoded? |
| Update policy | Are only queries updated, or both sides updated in alternating blocks? |

Cross-attention is directional. Ligand-to-pocket attention and pocket-to-ligand attention are not the same interface.

## Complexity

If the query length is $L_q$ and context length is $L_c$, the attention matrix has size:

$$
A\in\mathbb{R}^{L_q\times L_c}
$$

so the main memory and compute scale as:

$$
O(L_qL_c d)
$$

This matters for long documents, protein residues, dense image patches, and atom-level protein-ligand complexes.

## Leakage and Ground Truth Context

| Context Source | Risk |
| --- | --- |
| retrieved documents | answer appears verbatim in retrieval set |
| ground-truth structure | evaluation uses information unavailable at inference |
| ligand-defined pocket | pocket chosen using the answer ligand |
| future tokens | causal generation becomes invalid |
| tool output | tool may expose labels or hidden test information |

## Checks

- What object provides queries?
- What object provides keys and values?
- Is the context fixed, retrieved, generated, or ground truth?
- Does the model need bidirectional exchange, or one-way conditioning only?
- Are masks preventing access to future or invalid context?
- Does context length dominate the compute budget?
- Is cross-attention being interpreted as causal evidence without validation?

## Related

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder architectures]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[agents/index|Agents]]
