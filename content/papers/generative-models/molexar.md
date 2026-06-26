---
title: Molexar — Unified Multimodal Molecular Foundation Model for Drug Design
aliases:
  - papers/molexar
tags:
  - paper
  - molecular-generation
  - generative-models
  - structure-based-ai
status: reading
source_type: ArXiv
source_url: https://arxiv.org/abs/2606.25865
---

# Molexar: A Unified Multimodal Molecular Foundation Model for Drug Design

## One-line Summary

A unified multimodal molecular foundation model built on Fragment-SELFIES that achieves single autoregressive generation path across scalar properties, pharmacophore fingerprints, protein sequences, and binding pockets.

## Why It Matters

This paper may be relevant to [[research/structure-based-ai/index|Structure-based AI]] and [[concepts/generative-models/molecular-generation|molecular generation]]. The unified single-path design avoids per-condition separate heads and could simplify multi-property drug-design pipelines.

## Key Points

- Fragment-SELFIES: fragment-aware molecular language with validity-preserving decoding
- Single autoregressive decoder handles multiple condition types via in-place replacement of value-token embeddings
- Reported pretrained model behavior: 100% validity and high drug-likeness in unconditional and fragment-constrained generation
- Reported SFT behavior: single- and multi-property instruction following
- Reported target-conditioned generation on CrossDocked2020
- Reported MolGenBench safety and potency results; benchmark details remain to verify

## Reading Questions

- How does Fragment-SELFIES compare to standard SELFIES in terms of syntactic constraints and coverage?
- Does the in-place embedding replacement cause interference across condition types?
- What is the validity rate distribution across property types — does multi-property conditioning degrade single-property performance?
- How does the model scale with parameter count relative to specialized per-property models?
- Are the target-conditioned results robust to ligand scaffold split, protein-family split, and pocket similarity leakage?

## Artifact Availability

| Artifact | Status | Notes |
|---|---|---|
| Paper | found | arXiv abstract page |
| Code | to verify | Search project page or linked repository |
| Data | to verify | CrossDocked2020 and MolGenBench protocol details |
| Splits | to verify | Need target-conditioned generation split description |
| Weights | to verify | Public checkpoint status unknown |
| Environment | to verify | Implementation dependencies unknown |

## Related Notes

- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/fragment-selfies|Fragment-SELFIES]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[papers/reproducibility/artifact-availability|Artifact availability]]

## Metadata

- arXiv: [2606.25865](https://arxiv.org/abs/2606.25865)
- Submitted: 2026-06-24
- Authors: Haoyu Lin, Yiyan Liao, Jinmei Pan, Xinliao Ling, Luhua Lai, Jianfeng Pei
- Category: q-bio.BM
