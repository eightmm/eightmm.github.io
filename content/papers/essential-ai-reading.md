---
title: Essential AI Reading Queue
unlisted: true
tags:
  - papers
  - ai
  - reading-queue
---

# Essential AI Reading Queue

This page is a placeholder queue for papers that may deserve future notes. Keep entries short until a paper becomes important enough for a dedicated note.

## Architecture

| Topic | Candidate Papers | Route |
| --- | --- | --- |
| Transformer and attention | Attention Is All You Need, BERT, GPT-style decoder-only models | [[papers/architectures/index|Architecture papers]], [[papers/llm/index|LLM papers]] |
| Vision and convolution | AlexNet, ResNet, U-Net, Vision Transformer | [[papers/architectures/index|Architecture papers]] |
| Graph and geometry | GCN, GAT, SchNet, EGNN, SE(3)-Transformer, AlphaFold-style structure modules | [[papers/architectures/index|Architecture papers]], [[papers/computational-biology/index|Computational Biology papers]] |
| Sequence alternatives | S4, Mamba, linear attention variants | [[papers/architectures/index|Architecture papers]] |
| Conditional compute | Mixture of Experts, Switch Transformer, routed sparse models | [[papers/architectures/index|Architecture papers]], [[papers/systems/index|Systems papers]] |

## Learning Methods

| Topic | Candidate Papers | Route |
| --- | --- | --- |
| Self-supervised learning | SimCLR, MoCo, BYOL, MAE, DINO, JEPA-style objectives | [[papers/learning-methods/index|Learning method papers]] |
| Language-model pretraining | GPT, BERT, T5, instruction tuning, RLHF-style alignment | [[papers/llm/index|LLM papers]], [[papers/learning-methods/index|Learning method papers]] |
| Representation learning | contrastive learning, masked modeling, metric learning, probing papers | [[papers/learning-methods/index|Learning method papers]] |

## Generative Models

| Topic | Candidate Papers | Route |
| --- | --- | --- |
| Diffusion and score models | DDPM, score-based generative modeling, classifier-free guidance | [[papers/generative-models/index|Generative model papers]] |
| Flow and likelihood models | Normalizing flows, CNF, flow matching, rectified flow | [[papers/generative-models/index|Generative model papers]] |
| Structured generation | molecular generation, protein design, structure generation | [[papers/generative-models/index|Generative model papers]], [[papers/computational-biology/index|Computational Biology papers]] |

## Systems and Evaluation

| Topic | Candidate Papers | Route |
| --- | --- | --- |
| Scaling and efficiency | scaling laws, Chinchilla-style compute/data tradeoff, efficient inference | [[papers/systems/index|Systems papers]] |
| Evaluation and reproducibility | benchmark design, leakage audits, artifact papers | [[papers/reproducibility/index|Paper reproducibility]], [[papers/analysis/index|Paper analysis]] |
| Agents and tools | tool use, ReAct-style reasoning, retrieval, agent evaluation | [[papers/llm/index|LLM papers]], [[agents/index|Agents]] |

## Computational Biology

| Topic | Candidate Papers | Route |
| --- | --- | --- |
| Protein structure and design | AlphaFold family, RFdiffusion, protein language models | [[papers/computational-biology/index|Computational Biology papers]], [[papers/protein-modeling/index|Protein modeling papers]] |
| Structure-based modeling | DiffDock-style docking, PoseBusters-style evaluation, docking benchmarks | [[papers/sbdd/index|Structure-based modeling papers]] |
| Molecular modeling | molecular property prediction, molecular generation, assay-aware evaluation | [[papers/computational-biology/index|Computational Biology papers]] |

## Promotion Rule

Create a dedicated paper note only when at least one of these is true:

- The paper defines a concept used across several wiki pages.
- The paper is needed to interpret a current project or research direction.
- The paper is a baseline or evaluation reference for computational biology work.
- The paper changes how architecture, learning method, generation, systems, or evaluation should be explained.
