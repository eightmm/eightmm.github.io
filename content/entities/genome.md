---
title: Genome
tags:
  - entities
  - genome
  - sequence
---

# Genome

A genome is the complete genetic sequence of an organism or sample. In this wiki, genome notes are limited to sequence modeling and bio-AI context, not broad clinical omics or systems biology.

## Modeling Views

- Sequence view for tokenization, long-context modeling, and masked objectives.
- Region view for genes, regulatory elements, variants, and annotations.
- Dataset view for species, assemblies, quality control, and split design.

## Checks

- What sequence unit is modeled: base, k-mer, region, gene, or chromosome segment?
- Are species, samples, and near-duplicate regions separated across splits?
- Is the task representation learning, variant effect prediction, annotation, or generation?
- Are annotations and labels public, reproducible, and safe to cite?

## Related

- [[entities/sequence|Sequence]]
- [[entities/dataset|Dataset]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
