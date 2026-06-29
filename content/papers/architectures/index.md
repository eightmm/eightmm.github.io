---
title: Architecture Papers
tags:
  - papers
  - architectures
  - ai
---

# Architecture Papers

Architecture paper note는 attention, convolution, recurrence, graph neural network, state-space model, Mamba-style selective recurrence, mixture of experts, set model, geometric architecture 같은 model family와 structural design choice를 다룹니다.

논문의 오래 남는 기여가 재사용 가능한 architecture나 block이면 이 묶음에 둡니다. 같은 논문이 LLM 역사에서도 중요하면 노트를 복제하지 말고 [[papers/llm/index|LLM papers]]로 cross-link합니다.

## Reading Axes

- 어떤 input structure를 가정하는가: sequence, image, graph, set, 3D coordinate, multimodal context, agent state?
- 어떤 inductive bias가 바뀌는가: locality, permutation behavior, equivariance, recurrence, memory, sparsity, routing, hierarchy?
- sequence length, graph size, atom count, residue count, token count에 따라 complexity가 어떻게 바뀌는가?
- contribution이 new architecture, block replacement, scaling rule, efficiency trick 중 무엇인가?
- evidence가 accuracy, sample quality, transfer, stability, latency, memory, throughput 중 무엇에 관한 것인가?
- ablation이 objective, data, compute 변화와 architecture 변화를 분리할 만큼 강한가?

## Concepts

- [[ai/architectures|Architectures gateway]]
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
- [[papers/llm/index|LLM papers]]
- [[papers/workflows/ai-molecular-math-paper-template|AI-Molecular-Math paper template]]
- [[papers/analysis/ablation-map|Ablation map]]
- [[papers/analysis/paper-comparison-matrix|Paper comparison matrix]]
