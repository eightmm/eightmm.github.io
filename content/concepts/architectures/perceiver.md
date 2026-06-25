---
title: Perceiver
tags:
  - architectures
  - attention
  - multimodal
---

# Perceiver

Perceiver-style architectures use a latent array to read from large or heterogeneous inputs through cross-attention. They are useful as a pattern for multimodal inputs, long inputs, or inputs whose raw element count is expensive for full self-attention.

Given input tokens $X\in\mathbb{R}^{N\times d}$ and latent tokens $Z\in\mathbb{R}^{M\times d}$ with $M \ll N$, a latent cross-attention step is:

$$
Q = ZW_Q,\qquad K = XW_K,\qquad V = XW_V
$$

$$
Z' = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

The latent tokens can then be processed by self-attention or other blocks at cost tied to $M$ rather than $N$.

## Key Ideas

- Cross-attention compresses large inputs into a fixed-size latent workspace.
- The pattern separates input size from the cost of deeper latent processing.
- It can combine image patches, audio frames, text tokens, retrieved chunks, or structured features if they are embedded into compatible tokens.
- The bottleneck is useful but can discard information if $M$ is too small or training does not force the right summaries.

## Practical Checks

- Check what the latent array represents and whether it is learned, data-dependent, or task-specific.
- Track the input tokenization for each modality.
- Inspect whether output queries decode token-level, set-level, or sequence-level predictions.
- Compare with [[concepts/architectures/encoder-decoder|encoder-decoder]] and [[concepts/architectures/set-transformer|Set Transformer]] patterns.

## Related

- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/architectures/set-transformer|Set Transformer]]
