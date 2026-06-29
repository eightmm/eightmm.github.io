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

- What input structure is assumed: sequence, image, graph, set, 3D coordinate, multimodal context, or agent state?
- What inductive bias changes: locality, permutation behavior, equivariance, recurrence, memory, sparsity, routing, or hierarchy?
- What complexity changes with sequence length, graph size, atom count, residue count, or token count?
- Is the contribution a new architecture, a block replacement, a scaling rule, or an efficiency trick?
- Is the evidence about accuracy, sample quality, transfer, stability, latency, memory, or throughput?
- Are ablations strong enough to isolate the architecture from objective, data, or compute changes?

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

- Architecture gains may come from more parameters, tokens, data, training steps, or tuning budget.
- A new block may be tested with a different objective or augmentation policy.
- Throughput gains may depend on hardware, kernel, batch size, sequence length, or sparse implementation.
- Graph and geometric architectures can leak information through graph construction, coordinate frame choices, or template-derived features.

## Related

- [[ai/paper-intake|AI paper intake]]
- [[concepts/coverage-matrix|Coverage matrix]]
- [[papers/workflows/paper-triage|Paper triage]]
- [[papers/llm/index|LLM papers]]
- [[papers/workflows/ai-molecular-math-paper-template|AI-Molecular-Math paper template]]
- [[papers/analysis/ablation-map|Ablation map]]
- [[papers/analysis/paper-comparison-matrix|Paper comparison matrix]]
