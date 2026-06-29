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

For deeper latent processing:

$$
Z^{(\ell+1)}
=
\operatorname{SelfAttn}(Z^{(\ell)})
$$

and task outputs are usually decoded by pooling, linear readout, or output queries:

$$
\hat{y}=r_\psi(Z^{(L)})
$$

## Key Ideas

- Cross-attention compresses large inputs into a fixed-size latent workspace.
- The pattern separates input size from the cost of deeper latent processing.
- It can combine image patches, audio frames, text tokens, retrieved chunks, or structured features if they are embedded into compatible tokens.
- The bottleneck is useful but can discard information if $M$ is too small or training does not force the right summaries.

## When To Use

| Situation | Why Perceiver-style latents help |
| --- | --- |
| very large input token count | deep processing cost depends more on latent count $M$ than raw input count $N$ |
| heterogeneous modalities | each modality can be embedded into tokens before cross-attention |
| set-like or unordered inputs | latent array can summarize without requiring a natural sequence order |
| expensive pairwise attention | full $N^2$ self-attention can be replaced by input-to-latent attention plus latent processing |

## Cost Shape

One input-to-latent cross-attention layer has approximate attention cost:

$$
O(MNd_k)
$$

Latent self-attention then costs:

$$
O(M^2d_k)
$$

This is useful when $M \ll N$. If the task needs dense output over all input elements, decoding can reintroduce a large cost, so the output contract matters.

## Failure Modes

| Failure | Cause |
| --- | --- |
| information bottleneck | too few latents or weak objective for preserving rare details |
| modality imbalance | one modality dominates cross-attention because scale or token count is mismatched |
| weak interpretability | latent slots do not automatically correspond to human-readable objects |
| expensive decoding | dense per-token output requires attention back to many input positions |

## Practical Checks

- Check what the latent array represents and whether it is learned, data-dependent, or task-specific.
- Track the input tokenization for each modality.
- Inspect whether output queries decode token-level, set-level, or sequence-level predictions.
- Compare $N$, $M$, head dimension, and decoding cost before claiming scalability.
- Check whether each modality has normalization, positional features, or type embeddings.
- Compare with [[concepts/architectures/encoder-decoder|encoder-decoder]] and [[concepts/architectures/set-transformer|Set Transformer]] patterns.

## Related

- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/architectures/set-transformer|Set Transformer]]
