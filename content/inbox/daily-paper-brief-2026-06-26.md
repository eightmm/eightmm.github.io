---
title: Daily Paper Brief - 2026-06-26
date: 2026-06-26
tags:
  - daily-paper-brief
  - molecular
  - protein-modeling
  - generative-models
status: inbox
source: OpenClaw
---

# Daily Paper Brief - 2026-06-26

## Summary

- Top 5 papers selected
- Molecular / Protein ML: 3
- Pure-AI primitive: 2
- Notable signal: Unified multimodal molecular foundation model (Molexar) reaches 100% validity + strong drug-likeness; equivariant transformer backbone (MEET) achieves linear memory scaling for atomistic peptide design
- Public status: selected molecular/protein items were promoted to reading notes; pure-AI items remain in [[inbox/curation-queue|Curation queue]]

## Papers

### Molexar: A Unified Multimodal Molecular Foundation Model for Drug Design

- Track: [[research/structure-based-ai/index|Structure-based AI]]
- Paper note: [[papers/generative-models/molexar|Molexar]]
- Source metadata: arXiv:2606.25865, submitted 2026-06-24
- Related concepts:
  - [[concepts/generative-models/molecular-generation|Molecular generation]]
  - [[concepts/molecular-modeling/smiles|SMILES]]
  - [[concepts/molecular-modeling/fragment-selfies|Fragment-SELFIES]]
- Priority: Follow-up
- Score: 5.7 / 10
- Follow-up:
  - Verify MolGenBench benchmark details
  - Check cross-docked2020 baseline comparisons
  - Reproducibility: code release to verify

### Scalable Peptide Design via Memory-Efficient Equivariant Transformer (MEET)

- Track: [[research/protein-modeling/index|Protein modeling]]
- Paper note: [[papers/protein-modeling/meet-equivariant-peptide|MEET]]
- Source metadata: arXiv:2606.25006, submitted 2026-06-23
- Related concepts:
  - [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
  - [[concepts/generative-models/protein-design|Protein design]]
- Priority: Follow-up
- Score: 5.5 / 10
- Follow-up:
  - Verify AFDB dataset scale and benchmark results
  - Check linear memory scaling claims with atom count
  - Reproducibility: preprint or code to verify

### Multi-Scale Machine Learning for Antibody-Antigen Binding Affinity Prediction

- Track: [[research/structure-based-ai/index|Structure-based AI]]
- Paper note: [[papers/protein-modeling/multi-scale-antibody-binding|Multi-scale ML for Antibody-Antigen Binding]]
- Source metadata: bioRxiv preprint, posted 2026-06-23
- Related concepts:
  - [[concepts/sbdd/binding-affinity|Binding affinity]]
  - [[concepts/protein-modeling/protein-representation|Protein representation]]
  - [[concepts/evaluation/boltzmann-ceiling|Boltzmann ceiling analysis]]
- Priority: Hold
- Score: 5.34 / 10
- Follow-up:
  - Verify LOCO-DMS cross-validation methodology
  - Check Boltzmann ceiling analysis details
  - Cross-pathogen transfer failure: follow up for generalization research

### The Structural Origin of Attention Sink

- Track: AI
- Related concepts:
  - [[concepts/architectures/attention|Attention]]
  - [[concepts/architectures/transformer|Transformer]]
- Priority: Hold
- Score: 4.54 / 10
- Follow-up:
  - Verify venue and publication status
  - Check super neuron analysis details

### RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse

- Track: AI systems
- Related concepts:
  - [[concepts/systems/inference|Inference]]
  - [[concepts/llm/context-window|Context window]]
- Priority: Skip
- Score: 4.0 / 10
- Follow-up:
  - Verify 4.7x TTFT speedup claims
  - Check KV cache reuse rate benchmarks
