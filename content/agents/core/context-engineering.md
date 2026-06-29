---
title: Context Engineering
tags:
  - agents
  - llm
  - context
---

# Context Engineering

Context engineering은 agent에게 적절한 정보를 적절한 시점에 주는 실천입니다. Context가 너무 적으면 blind guess가 생기고, 너무 많으면 distraction, stale assumption, missed constraint가 생깁니다.

Working context는 아래처럼 볼 수 있습니다.

$$
C_t
=
G
\cup S_t
\cup E_t
\cup R_t
$$

여기서 $G$는 goal, $S_t$는 current working state, $E_t$는 tool 또는 file에서 얻은 evidence, $R_t$는 retrieved memory 또는 reference입니다.

## 원칙

- durable rule은 반복 prompt가 아니라 stable document에 둡니다.
- 모든 것을 load하기보다 필요할 때 evidence를 retrieve합니다.
- memory보다 primary file, command output, rendered artifact를 우선합니다.
- summary에는 uncertainty와 missing evidence를 명시합니다.
- shared context에서 secret과 private operational detail을 제거합니다.

## 확인할 것

- context가 actual current state를 포함하는가?
- stale summary가 fresh evidence로 override되는가?
- action 전에 constraint와 success criteria가 visible한가?
- untrusted document가 instruction과 분리되어 있는가?

## Related

- [[concepts/llm/context-packing|Context packing]]
- [[concepts/llm/chunking|Chunking]]
- [[concepts/llm/query-rewriting|Query rewriting]]
- [[concepts/llm/token-budget|Token budget]]
- [[concepts/llm/context-window|Context window]]
- [[concepts/llm/in-context-learning|In-context learning]]
- [[agents/core/agent-memory|Agent memory]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/verification/prompt-injection|Prompt injection]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
