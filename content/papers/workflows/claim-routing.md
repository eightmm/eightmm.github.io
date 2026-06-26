---
title: Claim Routing
unlisted: true
tags:
  - papers
  - workflows
  - routing
---

# Claim Routing

Claim routing decides which axis should own a new paper note or synthesis post. A paper can touch many areas, but the durable note should follow the strongest claim.

$$
\text{route}(p)
=
\arg\max_a \ \text{claim strength}(p,a)
$$

where $p$ is a paper or post candidate and $a$ is an axis such as architecture, learning method, generative modeling, molecular modeling, math, evaluation, systems, or agents.

## First Question

| Question | If Yes | Start |
| --- | --- | --- |
| Is the main contribution a model structure? | architecture paper or concept update | [Architecture papers](/papers/architectures) |
| Is the main contribution a supervision or training signal? | learning-method paper or concept update | [Learning method papers](/papers/learning-methods) |
| Is the main contribution sampling or density modeling? | generative-model paper or concept update | [Generative model papers](/papers/generative-models) |
| Is the main contribution about molecules, proteins, docking, conformers, or complexes? | molecular modeling paper intake | [Molecular modeling paper intake](/molecular-modeling/paper-intake) |
| Is the main contribution an equation, estimator, metric, or derivation? | formula intake or math concept update | [Formula intake](/math/formula-intake) |
| Is the main contribution a benchmark or evaluation protocol? | benchmark card or evaluation concept update | [Benchmark intake](/concepts/data/benchmark-intake) |
| Is the main contribution training, inference, reproducibility, or tooling? | systems or infra note | [Systems](/concepts/systems) |
| Is the main contribution tool-using LLM workflow behavior? | agent note | [Agents](/agents) |

## Tie Breakers

| Tie | Prefer | Reason |
| --- | --- | --- |
| Architecture vs learning method | learning method | if the same architecture could use the signal, the reusable idea is the objective |
| Architecture vs systems | systems | if speed or memory is the claim, the system boundary owns the evidence |
| Generative model vs molecular modeling | molecular modeling | if validity, docking, pose, affinity, or protein-ligand evaluation is central |
| Math vs AI method | AI method | if the formula is only the implementation of a model family |
| Math vs evaluation | evaluation | if the formula defines the reported metric or statistical comparison |
| Paper note vs concept update | concept update | if no paper-specific claim needs durable tracking |
| Inbox vs paper note | inbox | if metadata, public source, or relevance is still `to verify` |

## Claim Strength

Score each axis with a short note before creating a new paper page.

| Axis | Evidence That It Owns The Paper |
| --- | --- |
| Object | new entity, modality, representation contract, or preprocessing rule |
| Architecture | new block, connectivity pattern, inductive bias, scaling rule, or complexity change |
| Learning method | new target, masking rule, contrast set, pretraining signal, preference signal, or adaptation protocol |
| Generative model | new likelihood, score, velocity, latent path, sampler, or validity filter |
| Molecular modeling | molecule/protein/complex object boundary, docking workflow, conformer policy, label semantics, split, or leakage issue |
| Math | new estimator, derivation, objective decomposition, metric definition, or uncertainty calculation |
| Evaluation | new benchmark, split, metric, baseline, ablation, failure taxonomy, or reproducibility evidence |
| Systems | compute, memory, throughput, serving, reproducibility, artifact, scaling evidence, or implementation boundary |
| Agents | tool contract, planning loop, memory boundary, verifier, handoff, or orchestration protocol |

## Routing Output

For every promoted paper or synthesis post, record:

```text
primary axis: to verify
secondary axes: to verify
paper bucket: to verify
concept updates: to verify
concept update contract: to verify
benchmark/evaluation update: to verify
formula update: to verify
public artifact status: to verify
```

## Minimum Bundle By Axis

| Primary Axis | Minimum Companion Notes |
| --- | --- |
| Architecture | architecture concept, complexity or scaling note, ablation note, evaluation risk |
| Learning method | objective formula, data/signal description, transfer or evaluation protocol |
| Generative model | probability path or sampling formula, validity metrics, failure examples |
| Molecular modeling | object contract, preprocessing/split rule, leakage check, task metric |
| Math | symbol definitions, sampled distribution, relation to model or metric |
| Evaluation | benchmark contract, metric definition, selection rule, uncertainty |
| Systems | artifact availability, reproducibility checklist, compute or runtime boundary |
| Agents | tool contract, acceptance criteria, evidence ledger, verification loop |

## Related

- [[papers/workflows/paper-triage|Paper triage]]
- [[papers/workflows/ai-molecular-math-paper-template|AI-Molecular-Math paper template]]
- [[papers/workflows/concept-update-contract|Concept update contract]]
- [[concepts/coverage-matrix|Coverage matrix]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
- [[ai/paper-intake|AI paper intake]]
- [[molecular-modeling/paper-intake|Molecular modeling paper intake]]
- [[math/formula-intake|Formula intake]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/analysis/evidence-table|Evidence table]]
