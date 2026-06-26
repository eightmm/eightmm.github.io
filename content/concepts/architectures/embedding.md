---
title: Embedding
tags:
  - architectures
  - representation-learning
---

# Embedding

An embedding maps a discrete object or structured input into a continuous vector space. Tokens, residues, atoms, graph nodes, and retrieved chunks can all be embedded before model processing.

For a vocabulary item $i$, an embedding lookup is:

$$
e_i = E_i
$$

where $E\in\mathbb{R}^{V\times d}$ is an embedding table, $V$ is vocabulary size, and $d$ is embedding dimension.

For token sequences:

$$
x_t = e_{\mathrm{token}(t)} + p_t
$$

where $p_t$ is a positional or structural encoding.

Embeddings are often compared with [[concepts/math/vector-norm-similarity|dot product, cosine similarity, or Euclidean distance]], depending on whether magnitude should affect the score.

For a batch of token ids $X\in\{1,\ldots,V\}^{B\times T}$, lookup produces:

$$
H = E[X]\in\mathbb{R}^{B\times T\times d}
$$

An output classifier may tie weights to the input embedding table:

$$
\operatorname{logits}_t = h_t E^\top
$$

This reduces parameters and keeps input/output token spaces aligned.

## Common Embedding Types

- Token embeddings: wordpieces, residues, atoms, k-mers, patches, or graph nodes.
- Positional embeddings: sequence index, relative offset, chain id, or spatial bucket.
- Type embeddings: segment, modality, atom type, residue type, or role.
- Pretrained embeddings: frozen or fine-tuned representations from a larger model.

Embedding quality is not only dimensionality. It also depends on tokenization, normalization, pooling, masking, and whether the downstream metric matches the embedding geometry.

## Checks

- What does one token represent: wordpiece, residue, atom, k-mer, patch, or node?
- Are position, chain, segment, atom type, or edge features added separately?
- Is the embedding learned, fixed, pretrained, or tied to the output classifier?
- Does tokenization discard information needed by the task?

## Related

- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/math/vector-norm-similarity|Vector norm and similarity]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[entities/sequence|Sequence]]
