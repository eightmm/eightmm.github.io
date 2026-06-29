---
title: Coding Agents
tags:
  - agents
  - coding
  - research-engineering
---

# Coding Agents

Coding agent는 codebase를 inspect하고, file을 수정하고, command를 실행하고, 결과를 보고할 수 있는 LLM 기반 tool입니다. Task에 clear goal, local verification path, bounded blast radius가 있을 때 가장 유용합니다.

## 잘 맞는 사용처

- test가 있는 작은 module refactor.
- existing code를 바탕으로 documentation draft 작성.
- 알고 있는 codebase에 narrow feature 추가.
- 여러 file에 걸친 repetitive check 실행.
- commit 전 implementation risk review.

## 약한 지점

- vague product direction.
- hidden data 또는 environment assumption.
- explicit review 없는 security-sensitive change.
- written plan 없는 큰 dependency, API, schema, training change.

## Verification 습관

Agent가 도운 모든 change는 build, unit test, lint, smoke test, manual review 같은 concrete check로 끝나야 합니다. 중요한 질문은 agent가 confident하게 말했는지가 아니라 artifact가 correct한지입니다.

## Related

- [[agents/index|Agents]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[projects/index|Projects]]
- [[infra/index|Infra]]
