---
title: MEET — Memory-Efficient Equivariant Transformer for Scalable Peptide Design
tags:
  - paper
  - protein-design
  - geometric-deep-learning
  - equivariance
status: reading
source_type: ArXiv
source_url: https://arxiv.org/abs/2606.25006
---

# Scalable Peptide Design via Memory-Efficient Equivariant Transformer

## One-line Summary

An E(3)-equivariant backbone that achieves linear memory scaling with atom count via memory-efficient attention and sparse bond adaptation, integrated into a VAE + latent diffusion pipeline for full-atom peptide generation.

## Why It Matters

This paper may be relevant to [[research/protein-modeling/index|Protein modeling]] and [[concepts/geometric-deep-learning/equivariant-gnn|equivariant GNNs]]. The linear memory scaling claim is significant — if reproducible, it enables systematic model/data scaling for structure-based peptide design.

## Key Points

- MEET: E(3)-equivariant backbone with coupled invariant scalar and equivariant vector feature streams
- Memory-efficient attention reformulation of geometric computation
- Global coordinate aggregation for vector feature initialization
- Pairwise distances via augmented query/key dot products
- Sparse bond adaptation for covalent bond information
- Integrated into VAE + latent diffusion for full-atom peptide generation
- Reported linear memory scaling with atom count
- Reported improvements in binding affinity, physical validity, and sample diversity on AFDB-derived datasets; benchmark details remain to verify

## Reading Questions

- What is the concrete atom count where baseline equivariant models become memory-prohibitive?
- Does the memory-efficiency reformulation sacrifice geometric expressiveness (e.g., higher-order tensor interactions)?
- How does scaling behavior change with different peptide lengths (10 vs 50+ residues)?
- Is the improvement primarily from architecture or from the latent diffusion pipeline design?
- Does the evaluation isolate backbone scaling from VAE/diffusion training choices?
- Are peptide targets split by protein family, pocket similarity, and sequence identity?

## Artifact Availability

| Artifact | Status | Notes |
|---|---|---|
| Paper | found | arXiv abstract page |
| Code | to verify | Search project page or linked repository |
| Data | to verify | AFDB-derived filtering and task construction |
| Splits | to verify | Need target and peptide split policy |
| Weights | to verify | Public checkpoint status unknown |
| Environment | to verify | Implementation dependencies unknown |

## Related Notes

- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/generative-models/protein-design|Protein design]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/protein-modeling/protein-structure-cleaning|Protein structure cleaning]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[papers/artifact-availability|Artifact availability]]

## Metadata

- arXiv: [2606.25006](https://arxiv.org/abs/2606.25006)
- Submitted: 2026-06-23
- Authors: Rui Jiao, Xiangzhe Kong, Yinjun Jia, Yijia Zhang, Ziyi Yang, Yang Liu, Jianzhu Ma
- Category: cs.LG
