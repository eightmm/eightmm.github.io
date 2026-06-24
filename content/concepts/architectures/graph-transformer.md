---
title: Graph Transformers
tags:
  - architectures
  - graph-transformer
  - graphs
  - attention
---

# Graph Transformers

Graph transformers combine graph-structured inputs with attention-based message mixing. They are a bridge between [[concepts/architectures/gnn|GNNs]] and [[concepts/architectures/transformer|Transformers]].

## Key Ideas

- Attention can connect distant graph nodes without many local message-passing steps.
- Structural encodings tell the model about edges, distances, paths, centrality, or geometry.
- Some variants attend over all node pairs; others restrict attention to neighborhoods or sparse patterns.
- Edge or pair representations often sit beside node representations and carry relation-specific information.
- Geometry-aware graph transformers connect to [[concepts/geometric-deep-learning/index|geometric deep learning]] when coordinates, frames, or equivariant features are used.

## Practical Checks

- Identify whether graph edges only bias attention or also carry explicit messages.
- Check how positional, distance, or relation encodings enter the attention score and value update.
- Watch quadratic cost in atom, residue, or contact count.
- For protein-ligand complexes, check whether intra-protein, intra-ligand, and cross interactions are separated or shared.

## Related

- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
