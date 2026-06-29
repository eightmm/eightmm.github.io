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

- What is modeled: next token, masked token, instruction response, preference, reward, retrieval-augmented answer, or tool call?
- What changes: architecture, pretraining data, objective, context handling, alignment, retrieval, tool protocol, or evaluation?
- Is the paper about language modeling itself, a foundation model, an agent workflow, or an application benchmark?
- What evidence supports the claim: perplexity, downstream task score, human preference, factuality, tool success, latency, or cost?
- Are benchmark contamination, prompt sensitivity, decoding, and evaluation protocol controlled?

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
