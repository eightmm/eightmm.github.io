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

## Checks

- What does one token represent: wordpiece, residue, atom, k-mer, patch, or node?
- Are position, chain, segment, atom type, or edge features added separately?
- Is the embedding learned, fixed, pretrained, or tied to the output classifier?
- Does tokenization discard information needed by the task?

## Related

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[entities/sequence|Sequence]]
