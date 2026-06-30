---
title: 2026년 25주차 Molecular & Generative Modeling 읽기 노트
date: 2026-06-27
aliases:
  - posts/2026-06-27-week-25-molecular-generative-reading-notes
tags:
  - blog-post
  - protein-modeling
  - generative-models
  - molecular-generation
---

# 2026년 25주차 Molecular & Generative Modeling 읽기 노트

## 개요

이번 주 molecular generative AI와 protein modeling 쪽에서 눈에 띈 흐름은 아래와 같습니다.

- **Unified multimodal molecular generation**: fragment, property, pharmacophore, structural constraint를 하나의 generation path에서 다루는 single-autoregressive model이 늘고 있습니다. Condition별 special-purpose head에서 벗어나려는 흐름입니다.
- **Equivariant scaling**: linear memory scaling을 갖는 E(3)-equivariant transformer는 이전에는 다루기 어려웠던 크기의 full-atom peptide와 binder design 가능성을 넓힙니다.
- **Benchmark chasing보다 evaluation depth**: Boltzmann ceiling analysis와 confidence-stratified evaluation은 단순 score보다 protein interaction prediction에서 이론적 정보 한계가 어디인지 이해하는 쪽으로 초점을 옮깁니다.
- **Convergent folding computation**: ESMFold, OpenFold, Boltz-1 trunk를 causal tracing으로 비교하면 AI folding model이 정보를 처리하는 공통 two-stage structure가 보인다는 주장입니다. 공통 computational primitive가 있을 수 있습니다.
- **Survey signal**: Purdue dissertation 수준의 survey는 diffusion-based structure modeling, AlphaFold-Multimer distillation, trajectory generation(PathFold) 쪽으로 field가 정리되고 있음을 보여줍니다.

## 다룬 논문

- [[papers/generative-models/molexar|Molexar]]: Fragment-SELFIES와 single autoregressive path를 쓰는 unified multimodal molecular foundation model입니다. Unconditional generation에서 100% validity와 높은 drug-likeness를 보고합니다.
- [[papers/protein-modeling/meet-equivariant-peptide|MEET]]: scalable peptide design을 위한 memory-efficient equivariant transformer입니다. Atom 수에 대해 linear memory scaling을 주장합니다.
- [[papers/protein-modeling/multi-scale-antibody-binding|Multi-scale ML for Antibody-Antigen Binding]]: DMS feature를 사용한 Boltzmann ceiling analysis와 confidence-stratified prediction을 다룹니다. Cross-pathogen transfer의 한계를 보여줍니다.

## 보류한 논문

- **Two Stages of Folding**(arXiv 2602.06020, to verify): ESMFold, OpenFold, Boltz-1 trunk에 causal intervention을 적용해 convergent two-stage computation을 주장합니다. 점수는 괜찮지만 방법 확인이 더 필요합니다.
- **Deep Learning for Biomolecular Structure Modeling**(Purdue dissertation, 2026): CryoZeta diffusion, DistPepFold, PathFold trajectory generation을 다루는 literature-level survey입니다. Primary paper note보다는 reference로 유용합니다.
- **CARVE: Content-Aware Recurrent with Value Efficiency**(arXiv 2606.27229): 낮은 overhead로 transformer quality에 접근하는 linear attention 계열 주장입니다. 현재 domain fit은 낮아 pure-AI tracking 후보로 보류합니다.
- **Information-Aware KV Cache Compression for Long Reasoning**: forward influence metric을 통한 KV cache compression입니다. 현재는 molecular/protein 쪽보다 AI systems tracking 후보로 보류합니다.

## Concept update 후보

- [[concepts/generative-models/protein-design|Protein design]]: MEET는 equivariant VAE와 latent diffusion 조합이 full-atom peptide generation pipeline으로 쓰일 수 있음을 보여줍니다.
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]: efficient attention reformulation을 통한 linear memory scaling은 SE(3)-equivariant peptide design의 compute ceiling을 낮춥니다.
- [[concepts/generative-models/molecular-generation|Molecular generation]]: Fragment-SELFIES와 single-path autoregressive conditioning은 per-property head 중심 설계에서 벗어나는 방향입니다.
- [[concepts/sbdd/binding-affinity|Binding affinity]]: Boltzmann ceiling analysis는 antibody-antigen prediction에서 “remaining gap”이 무엇을 뜻하는지 해석하는 데 유용한 upper bound를 제공합니다.

## 다음에 볼 것

- 승격한 Molexar/MEET note를 직접 읽고 artifact availability와 code release 상태를 확인합니다.
- Fragment-SELFIES coverage를 확인하고 standard SELFIES와 비교합니다.
- Linear attention이 실제 관심 주제가 되면 CARVE의 formal theorem, Lyapunov stability, expressivity 주장을 추적합니다.
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]] 확장 여부를 봅니다. Two Stages of Folding의 convergent folding-trunk analysis는 방법론이 확인되면 subsection으로 키울 수 있습니다.
