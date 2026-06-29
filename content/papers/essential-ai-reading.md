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
| Transformer and attention | Attention Is All You Need, BERT, GPT-style decoder-only models | [Architecture papers](/papers/architectures), [LLM papers](/papers/llm) |
| Vision and convolution | AlexNet, ResNet, U-Net, Vision Transformer | [Architecture papers](/papers/architectures) |
| Graph and geometry | GCN, GAT, SchNet, EGNN, SE(3)-Transformer, AlphaFold-style structure modules | [Architecture papers](/papers/architectures), [Computational Biology papers](/papers/computational-biology) |
| Sequence alternatives | S4, Mamba, linear attention variants | [Architecture papers](/papers/architectures) |
| Conditional compute | Mixture of Experts, Switch Transformer, routed sparse models | [Architecture papers](/papers/architectures), [Systems papers](/papers/systems) |

## Learning Methods

| Topic | Candidate Papers | Route |
| --- | --- | --- |
| Self-supervised learning | SimCLR, MoCo, BYOL, MAE, DINO, JEPA-style objectives | [Learning method papers](/papers/learning-methods) |
| Language-model pretraining | GPT, BERT, T5, instruction tuning, RLHF-style alignment | [LLM papers](/papers/llm), [Learning method papers](/papers/learning-methods) |
| Representation learning | contrastive learning, masked modeling, metric learning, probing papers | [Learning method papers](/papers/learning-methods) |

## Generative Models

| Topic | Candidate Papers | Route |
| --- | --- | --- |
| Diffusion and score models | DDPM, score-based generative modeling, classifier-free guidance | [Generative model papers](/papers/generative-models) |
| Flow and likelihood models | Normalizing flows, CNF, flow matching, rectified flow | [Generative model papers](/papers/generative-models) |
| Structured generation | molecular generation, protein design, structure generation | [Generative model papers](/papers/generative-models), [Computational Biology papers](/papers/computational-biology) |

## Systems and Evaluation

| Topic | Candidate Papers | Route |
| --- | --- | --- |
| Scaling and efficiency | scaling laws, Chinchilla-style compute/data tradeoff, efficient inference | [Systems papers](/papers/systems) |
| Evaluation and reproducibility | benchmark design, leakage audits, artifact papers | [Paper reproducibility](/papers/reproducibility), [Paper analysis](/papers/analysis) |
| Agents and tools | tool use, ReAct-style reasoning, retrieval, agent evaluation | [LLM papers](/papers/llm), [Agents](/agents) |

## Computational Biology

| Topic | Candidate Papers | Route |
| --- | --- | --- |
| Protein structure and design | AlphaFold family, RFdiffusion, protein language models | [Computational Biology papers](/papers/computational-biology), [Protein modeling papers](/papers/protein-modeling) |
| Structure-based modeling | DiffDock-style docking, PoseBusters-style evaluation, docking benchmarks | [Structure-based modeling papers](/papers/sbdd) |
| Molecular modeling | molecular property prediction, molecular generation, assay-aware evaluation | [Computational Biology papers](/papers/computational-biology) |

## Promotion Rule

Create a dedicated paper note only when at least one of these is true:

- The paper defines a concept used across several wiki pages.
- The paper is needed to interpret a current project or research direction.
- The paper is a baseline or evaluation reference for computational biology work.
- The paper changes how architecture, learning method, generation, systems, or evaluation should be explained.
