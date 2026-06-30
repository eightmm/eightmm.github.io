---
title: Architecture Papers
tags:
  - papers
  - architectures
  - ai
---

# Architecture Papers

Architecture paper note는 오래 남는 model family, block, inductive bias, scaling route를 다룹니다. 이 페이지는 개별 논문을 모두 펼치는 곳이 아니라, architecture를 읽기 위한 paper shelf입니다.

논문의 오래 남는 기여가 재사용 가능한 architecture나 block이면 이 묶음에 둡니다. 같은 논문이 language-model history에서도 중요하면 노트를 복제하지 말고 [[concepts/llm/index|LLM concepts]]나 [[agents/index|Agents]]로 cross-link합니다.

## Shelves

| Shelf | Read For | Anchor Papers |
| --- | --- | --- |
| Sequence and attention | token mixing, position, long-range dependency, attention alternatives | [Attention Is All You Need](/papers/architectures/attention-is-all-you-need), [Layer Normalization](/papers/architectures/layer-normalization), [Mamba](/papers/architectures/mamba) |
| Vision backbones | locality, residual depth, dense prediction, patch tokenization, hierarchy | [AlexNet](/papers/architectures/alexnet), [Deep Residual Learning](/papers/architectures/deep-residual-learning), [U-Net](/papers/architectures/u-net), [Vision Transformer](/papers/architectures/vision-transformer), [Swin Transformer](/papers/architectures/swin-transformer) |
| Graphs and sets | permutation behavior, message passing, unordered inputs | [GCN](/papers/architectures/gcn), [Graph Attention Networks](/papers/architectures/graph-attention-networks), [Deep Sets](/papers/architectures/deep-sets), [Set Transformer](/papers/architectures/set-transformer) |
| Conditional compute | sparse routing, expert capacity, scaling under fixed token compute | [Switch Transformer](/papers/architectures/switch-transformer) |
| Training-time architecture blocks | normalization, activation scale, residual stability | [Batch Normalization](/papers/architectures/batch-normalization), [Layer Normalization](/papers/architectures/layer-normalization) |

## Current Notes

| Paper | Main Architecture | Why it is here |
| --- | --- | --- |
| [ImageNet Classification with Deep CNNs](/papers/architectures/alexnet) | AlexNet | large-scale deep CNN vision milestone |
| [Attention Is All You Need](/papers/architectures/attention-is-all-you-need) | Transformer | attention-only sequence transduction backbone |
| [Batch Normalization](/papers/architectures/batch-normalization) | BatchNorm | normalization as an architecture component |
| [Deep Residual Learning](/papers/architectures/deep-residual-learning) | ResNet | residual learning and deep CNN optimization |
| [U-Net](/papers/architectures/u-net) | U-Net | encoder-decoder CNN with localization skip paths |
| [An Image is Worth 16x16 Words](/papers/architectures/vision-transformer) | Vision Transformer | image patches as Transformer tokens |
| [Swin Transformer](/papers/architectures/swin-transformer) | Swin Transformer | shifted-window hierarchical vision Transformer |
| [Layer Normalization](/papers/architectures/layer-normalization) | LayerNorm | batch-independent normalization for sequence models |
| [Semi-Supervised Classification with GCNs](/papers/architectures/gcn) | GCN | graph message passing for node classification |
| [Graph Attention Networks](/papers/architectures/graph-attention-networks) | GAT | learned attention over graph neighborhoods |
| [Deep Sets](/papers/architectures/deep-sets) | Deep Sets | permutation-invariant set function architecture |
| [Set Transformer](/papers/architectures/set-transformer) | Set Transformer | attention-based permutation-invariant set modeling |
| [Mamba](/papers/architectures/mamba) | selective SSM | input-dependent state-space sequence modeling |
| [Switch Transformer](/papers/architectures/switch-transformer) | sparse MoE Transformer | top-1 expert routing and conditional compute |

## Reading Axes

- 어떤 input structure를 가정하는가: sequence, image, graph, set, 3D coordinate, multimodal context, agent state?
- 어떤 inductive bias가 바뀌는가: locality, permutation behavior, equivariance, recurrence, memory, sparsity, routing, hierarchy?
- sequence length, graph size, atom count, residue count, token count에 따라 complexity가 어떻게 바뀌는가?
- contribution이 new architecture, block replacement, scaling rule, efficiency trick 중 무엇인가?
- evidence가 accuracy, sample quality, transfer, stability, latency, memory, throughput 중 무엇에 관한 것인가?
- ablation이 objective, data, compute 변화와 architecture 변화를 분리할 만큼 강한가?

## Concepts

- [[ai/architectures|Architectures]]
- [[concepts/architectures/index|Architecture concepts]]
- [[concepts/architectures/architecture-selection|Architecture selection]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-transformer|Graph transformer]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/mamba|Mamba]]
- [[concepts/architectures/mixture-of-experts|Mixture of experts]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]

## Evaluation Risks

- Architecture gain이 parameter, token, data, training step, tuning budget 증가에서 온 것일 수 있습니다.
- New block이 다른 objective나 augmentation policy와 함께 test되었을 수 있습니다.
- Throughput gain은 hardware, kernel, batch size, sequence length, sparse implementation에 의존할 수 있습니다.
- Graph/geometric architecture는 graph construction, coordinate frame choice, template-derived feature를 통해 information을 leak할 수 있습니다.

## Related

- [[ai/paper-intake|AI paper intake]]
- [[concepts/coverage-matrix|Coverage matrix]]
- [[papers/workflows/paper-triage|Paper triage]]
- [[agents/index|Agents]]
- [[papers/workflows/ai-molecular-math-paper-template|AI-Molecular-Math paper template]]
- [[papers/analysis/ablation-map|Ablation map]]
- [[papers/analysis/paper-comparison-matrix|Paper comparison matrix]]
