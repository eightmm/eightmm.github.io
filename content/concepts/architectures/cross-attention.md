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

## Checks

- What object provides queries?
- What object provides keys and values?
- Is the context fixed, retrieved, generated, or ground truth?
- Does the model need bidirectional exchange, or one-way conditioning only?
- Are masks preventing access to future or invalid context?

## Related

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder architectures]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[agents/index|Agents]]
