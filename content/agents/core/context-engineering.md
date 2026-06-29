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

Context engineering은 더 많은 text를 넣는 문제가 아닙니다. 다음 action에 필요한 minimum sufficient context를 고르고, 오래된 요약과 최신 evidence를 분리하는 문제입니다.

## Context Source

| Source | 신뢰 방식 |
| --- | --- |
| User request | goal과 constraint의 primary source |
| Repository/file | current artifact state |
| Command output | build, test, status evidence |
| Retrieved memory | useful but stale 가능 |
| Web/source document | citation과 freshness 확인 필요 |
| Tool result | instruction이 아니라 observation |

## 원칙

- durable rule은 반복 prompt가 아니라 stable document에 둡니다.
- 모든 것을 load하기보다 필요할 때 evidence를 retrieve합니다.
- memory보다 primary file, command output, rendered artifact를 우선합니다.
- summary에는 uncertainty와 missing evidence를 명시합니다.
- shared context에서 secret과 private operational detail을 제거합니다.

## Failure Mode

- 오래된 summary가 current file보다 우선됩니다.
- 너무 많은 unrelated file이 들어와 constraint를 놓칩니다.
- tool output 안의 malicious instruction을 trusted instruction으로 취급합니다.
- private path, account, server detail이 shared context나 public note로 들어갑니다.
- context packing 때문에 acceptance criteria가 빠집니다.

## 확인할 것

- context가 actual current state를 포함하는가?
- stale summary가 fresh evidence로 override되는가?
- action 전에 constraint와 success criteria가 visible한가?
- untrusted document가 instruction과 분리되어 있는가?
- 다음 action에 필요 없는 context를 줄였는가?

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
