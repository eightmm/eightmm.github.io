---
title: LLM Papers
tags:
  - papers
  - llm
  - ai
---

# LLM Papers

LLM paper note는 language model, token prediction, scaling, instruction tuning, alignment, retrieval, tool use, agent-facing language-model workflow를 다룹니다.

중심 claim이 language-model behavior, training signal, context use, retrieval, alignment, tool-using workflow에 있으면 이 묶음에 둡니다. 중심 claim이 attention, recurrence, state-space model, mixture of experts처럼 재사용 가능한 model block이면 [[papers/architectures/index|Architecture papers]]를 씁니다.

## Reading Axes

- 무엇을 모델링하는가: next token, masked token, instruction response, preference, reward, retrieval-augmented answer, tool call?
- 무엇이 바뀌는가: architecture, pretraining data, objective, context handling, alignment, retrieval, tool protocol, evaluation?
- paper가 language modeling 자체, foundation model, agent workflow, application benchmark 중 무엇에 관한 것인가?
- claim을 지지하는 evidence가 perplexity, downstream task score, human preference, factuality, tool success, latency, cost 중 무엇인가?
- benchmark contamination, prompt sensitivity, decoding, evaluation protocol이 control되어 있는가?

## Concepts

- [[concepts/llm/index|LLM concepts]]
- [[concepts/llm/context-window|Context window]]
- [[concepts/llm/decoding|Decoding]]
- [[concepts/llm/in-context-learning|In-context learning]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/evidence-grounded-generation|Evidence-grounded generation]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/learning/instruction-tuning|Instruction tuning]]
- [[concepts/learning/preference-optimization|Preference optimization]]
- [[agents/index|Agents]]

## Curated Notes

아직 선별된 LLM paper note는 없습니다. Attention이나 Transformer처럼 foundational architecture paper는 [[papers/architectures/index|Architecture papers]]에서 시작하고, LLM reading path가 중요할 때 여기로 cross-link합니다.

## Related

- [[ai/index|AI]]
- [[ai/architectures|Architectures]]
- [[agents/index|Agents]]
- [[papers/workflows/longform-paper-review-guide|Longform paper review guide]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/analysis/paper-comparison-matrix|Paper comparison matrix]]
