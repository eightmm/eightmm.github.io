---
title: Genome
tags:
  - entities
  - genome
  - sequence
---

# Genome

A genome is the complete genetic sequence of an organism or sample. In this wiki, genome notes are limited to sequence modeling and Molecular Modeling context, not broad clinical omics or systems biology.

The local scope is intentionally narrow:

$$
\text{genome notes}
\subset
\{\text{sequence modeling},
\text{genomic regions},
\text{variant effect prediction},
\text{annotation tasks}\}
$$

It does not open broad single-cell, transcriptomics, pathway, or clinical-omics coverage unless the blog scope is explicitly expanded.

## Modeling Views

- Sequence view for tokenization, long-context modeling, and masked objectives.
- Region view for genes, regulatory elements, variants, and annotations.
- Dataset view for species, assemblies, quality control, and split design.

## Unit of Modeling

Genome tasks need a clear unit:

$$
u
\in
\{\text{base},
\text{k-mer},
\text{region},
\text{gene},
\text{variant},
\text{chromosome segment}\}
$$

The output may be a token prediction, annotation label, variant effect score, retrieval result, or generated sequence. The split should match that unit rather than arbitrary rows.

## Leakage and Provenance

Genome-scale models can leak through near-duplicate regions, shared assemblies, overlapping windows, or annotations derived from the same evidence source:

$$
\operatorname{overlap}(r_{\mathrm{train}}, r_{\mathrm{test}})
=
0
$$

when evaluating region-level generalization. For species or family generalization, the grouping key should be species, clade, sample, or region family as appropriate.

## Checks

- What sequence unit is modeled: base, k-mer, region, gene, or chromosome segment?
- Are species, samples, and near-duplicate regions separated across splits?
- Is the task representation learning, variant effect prediction, annotation, or generation?
- Are annotations and labels public, reproducible, and safe to cite?
- Which genome assembly, annotation version, and coordinate convention are used?
- Do train and test windows overlap directly or through reverse complements?
- Is the claim within-species, cross-species, cross-assembly, or variant-level generalization?

## Related

- [[concepts/genome-modeling/index|Genome modeling concepts]]
- [[concepts/genome-modeling/genomic-region|Genomic region]]
- [[concepts/genome-modeling/k-mer|K-mer]]
- [[concepts/genome-modeling/variant-effect-prediction|Variant effect prediction]]
- [[entities/sequence|Sequence]]
- [[entities/dataset|Dataset]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
