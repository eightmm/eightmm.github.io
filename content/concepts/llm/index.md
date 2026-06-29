---
title: LLM Concepts
tags:
  - llm
  - concepts
---

# LLM Concepts

LLM concept는 agent와 wiki-style knowledge base를 지탱하는 language model, context, retrieval, workflow pattern을 설명합니다.

LLM note는 "모델이 텍스트를 생성한다"에서 멈추지 않고, 어떤 context와 constraint 아래에서 어떤 output을 만들고 어떻게 검증하는지 봅니다.

$$
\hat{y}
\sim
p_\theta(y \mid I, C, T, F)
$$

where $I$ is instruction, $C$ is context, $T$ is optional tool or retrieval evidence, and $F$ is output format or constraint.

## Route Map

| Question | Start | Main Risk |
| --- | --- | --- |
| What is the model optimizing? | [Language model](/concepts/llm/language-model), [Autoregressive model](/concepts/generative-models/autoregressive-model) | likelihood treated as truth |
| What can fit in context? | [Context window](/concepts/llm/context-window), [Token budget](/concepts/llm/token-budget), [Context packing](/concepts/llm/context-packing) | missing or stale evidence |
| How should I ask? | [Prompting](/concepts/llm/prompting), [In-context learning](/concepts/llm/in-context-learning) | prompt style confused with verification |
| How is output sampled? | [Decoding](/concepts/llm/decoding) | temperature and sampling budget hidden |
| How is output constrained? | [Structured output](/concepts/llm/structured-output), [Tool calling](/concepts/llm/tool-calling) | valid syntax but wrong semantics |
| How is evidence retrieved? | [Retrieval-augmented generation](/concepts/llm/retrieval-augmented-generation), [Embedding retrieval](/concepts/llm/embedding-retrieval), [Hybrid retrieval](/concepts/llm/hybrid-retrieval) | retrieved text treated as instruction |
| How are claims grounded? | [Hallucination and grounding](/concepts/llm/hallucination-grounding), [Evidence-grounded generation](/concepts/llm/evidence-grounded-generation), [Citation grounding](/concepts/llm/citation-grounding) | citation does not support exact claim |
| When does this become an agent? | [Tool calling](/concepts/llm/tool-calling), [Agents](/agents), [Agent loop](/agents/core/agent-loop) | side effects without verification |

## Core Concepts

| Group | Notes |
| --- | --- |
| Model basics | [Language model](/concepts/llm/language-model), [Decoding](/concepts/llm/decoding), [In-context learning](/concepts/llm/in-context-learning) |
| Context | [Context window](/concepts/llm/context-window), [Token budget](/concepts/llm/token-budget), [Context packing](/concepts/llm/context-packing), [Prompting](/concepts/llm/prompting) |
| Retrieval | [Retrieval-augmented generation](/concepts/llm/retrieval-augmented-generation), [Embedding retrieval](/concepts/llm/embedding-retrieval), [Chunking](/concepts/llm/chunking), [Hybrid retrieval](/concepts/llm/hybrid-retrieval), [Query rewriting](/concepts/llm/query-rewriting) |
| Output and tools | [Structured output](/concepts/llm/structured-output), [Tool calling](/concepts/llm/tool-calling), [Inference contract](/concepts/systems/inference-contract) |
| Grounding and safety | [Prompt injection boundary](/concepts/llm/prompt-injection-boundary), [Hallucination and grounding](/concepts/llm/hallucination-grounding), [Evidence-grounded generation](/concepts/llm/evidence-grounded-generation), [Citation grounding](/concepts/llm/citation-grounding) |
| Compression and transfer | [Knowledge distillation](/concepts/learning/knowledge-distillation), [Model card](/concepts/systems/model-card) |

## LLM vs Agent Boundary

| If the note is about | Put it under |
| --- | --- |
| probability of text, decoding, context, prompt, retrieval, grounding | LLM concepts |
| tool contract, action loop, memory, planning, workflow completion | Agents |
| serving, latency, model card, inference contract | AI Systems |
| output metric, hallucination audit, claim evidence | Evaluation or Agents verification |

The boundary is action. A model that proposes a tool call is still an LLM concept; a workflow that executes tools, observes results, updates state, and verifies completion belongs under [[agents/index|Agents]].

$$
\text{LLM}
:
x \rightarrow y
$$

$$
\text{Agent}
:
x \rightarrow a_t \rightarrow o_t \rightarrow s_{t+1}
$$

## Claim Types

| Claim | Evidence |
| --- | --- |
| model can answer from context | context contains the evidence and answer cites it correctly |
| RAG improves factuality | retrieval evaluation plus answer-level grounding check |
| prompt improves behavior | fixed task set, baseline prompt, output rubric |
| structured output is reliable | schema validity and semantic validation |
| tool calling works | tool result handling, side-effect boundary, completion audit |
| agent workflow succeeds | state/action trace and external verification |

## Checks

- Is the model being used for generation, classification, extraction, retrieval, or tool orchestration?
- What context is provided, and what evidence is missing?
- How is token budget allocated?
- What decoding and output constraints are used?
- Are retrieved documents trusted as data, not instructions?
- Is retrieval unit, query rewriting, and reranking behavior visible enough to debug?
- Are generated claims grounded in evidence or marked `to verify`?
- Is the output verified outside the model?
- Is the task actually LLM-only, or does it need an agent/tool/workflow note?
- Does the claim distinguish language-model behavior from product feature behavior?
- Does the note specify context source, output format, and verification boundary?

## Related

- [[agents/index|Agents]]
- [[agents/core/agent-architecture|Agent architecture]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[agents/core/context-engineering|Context engineering]]
- [[concepts/systems/inference-contract|Inference contract]]
- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
