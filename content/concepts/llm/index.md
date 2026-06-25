---
title: LLM Concepts
tags:
  - llm
  - concepts
---

# LLM Concepts

LLM concepts describe language models, context, retrieval, and workflow patterns that support agents and wiki-style knowledge bases.

## Core Concepts

- [[concepts/llm/language-model|Language model]]
- [[concepts/llm/context-window|Context window]]
- [[concepts/llm/token-budget|Token budget]]
- [[concepts/llm/context-packing|Context packing]]
- [[concepts/llm/prompting|Prompting]]
- [[concepts/llm/in-context-learning|In-context learning]]
- [[concepts/llm/decoding|Decoding]]
- [[concepts/llm/structured-output|Structured output]]
- [[concepts/llm/tool-calling|Tool calling]]
- [[concepts/systems/inference-contract|Inference contract]]
- [[concepts/systems/model-card|Model card]]
- [[concepts/llm/prompt-injection-boundary|Prompt injection boundary]]
- [[concepts/llm/hallucination-grounding|Hallucination and grounding]]
- [[concepts/llm/evidence-grounded-generation|Evidence-grounded generation]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[concepts/learning/knowledge-distillation|Knowledge distillation]]

## Checks

- Is the model being used for generation, classification, extraction, retrieval, or tool orchestration?
- What context is provided, and what evidence is missing?
- How is token budget allocated?
- What decoding and output constraints are used?
- Are retrieved documents trusted as data, not instructions?
- Are generated claims grounded in evidence or marked `to verify`?
- Is the output verified outside the model?

## Related

- [[agents/index|Agents]]
- [[agents/core/agent-architecture|Agent architecture]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[agents/core/context-engineering|Context engineering]]
- [[concepts/systems/inference-contract|Inference contract]]
- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
