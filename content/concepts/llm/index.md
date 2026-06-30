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

## LLM Contract

LLM note는 prompt trick이 아니라, 입력과 증거와 출력 제약을 분리한 contract로 써야 합니다.

$$
\mathcal{L}_{\mathrm{LLM}}
=
(I,\ C,\ R,\ D,\ F,\ V)
$$

| Part | Meaning | Typical question |
| --- | --- | --- |
| $I$ | instruction | 무엇을 하라고 했는가? |
| $C$ | context | 모델이 볼 수 있는 user data, files, history, examples는 무엇인가? |
| $R$ | retrieved evidence | 검색된 문서는 어떤 query, chunk, ranking으로 왔는가? |
| $D$ | decoding policy | temperature, top-p, beam, sampling budget은 무엇인가? |
| $F$ | output format | free text, markdown, JSON schema, citation, tool call 중 무엇인가? |
| $V$ | verifier | 출력이 맞다는 외부 증거는 무엇인가? |

This makes a useful separation:

$$
\text{fluent answer}
\neq
\text{grounded answer}
\neq
\text{verified answer}
$$

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

## Context-Evidence Pipeline

For LLM Wiki writing, the important path is not only token generation. It is evidence selection and claim verification.

$$
\text{source}
\rightarrow
\text{chunk}
\rightarrow
\text{retrieve}
\rightarrow
\text{pack context}
\rightarrow
\text{generate}
\rightarrow
\text{verify}
$$

| Stage | Note | Failure mode |
| --- | --- | --- |
| Source | [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]] | stale, private, or irrelevant source |
| Chunk | [[concepts/llm/chunking|Chunking]] | chunk breaks the evidence needed for a claim |
| Retrieve | [[concepts/llm/embedding-retrieval|Embedding retrieval]], [[concepts/llm/hybrid-retrieval|Hybrid retrieval]] | high similarity but wrong evidence |
| Rewrite | [[concepts/llm/query-rewriting|Query rewriting]] | query changes the user's intent |
| Pack | [[concepts/llm/context-packing|Context packing]], [[concepts/llm/token-budget|Token budget]] | important evidence is excluded |
| Generate | [[concepts/llm/decoding|Decoding]], [[concepts/llm/prompting|Prompting]] | fluent unsupported claim |
| Verify | [[concepts/llm/evidence-grounded-generation|Evidence-grounded generation]], [[concepts/llm/citation-grounding|Citation grounding]] | citation does not entail the answer |

## RAG and Grounding Template

RAG note should preserve enough detail to debug retrieval and generation separately.

| Field | Write |
| --- | --- |
| Question | exact information need or task |
| Source set | document collection, freshness, public/private boundary |
| Unit | page, paragraph, section, chunk, table, code block |
| Retrieval | sparse, dense, hybrid, reranking, query rewrite |
| Context packing | what was included, truncated, summarized, or excluded |
| Generation | output format, citation style, decoding policy |
| Verification | evidence supports exact claim, not just related topic |
| Failure slices | missing source, stale source, conflicting source, unsupported citation |

## Claim Types

| Claim | Evidence |
| --- | --- |
| model can answer from context | context contains the evidence and answer cites it correctly |
| RAG improves factuality | retrieval evaluation plus answer-level grounding check |
| prompt improves behavior | fixed task set, baseline prompt, output rubric |
| structured output is reliable | schema validity and semantic validation |
| tool calling works | tool result handling, side-effect boundary, completion audit |
| agent workflow succeeds | state/action trace and external verification |

## Common Failure Modes

| Failure | Why it matters |
| --- | --- |
| treating context as truth | provided text can be stale, wrong, or adversarial |
| treating retrieved text as instruction | retrieval content can override the intended task if not bounded |
| citing a related source | citation may not support the exact claim |
| hiding decoding settings | sampling variance makes behavior hard to compare |
| using prompt examples as evaluation | prompt demonstration is not a held-out test |
| confusing product feature with model concept | UI behavior, memory, tools, and model capability are different layers |
| moving side effects into an LLM-only note | once tools execute, the topic belongs closer to Agents |

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
