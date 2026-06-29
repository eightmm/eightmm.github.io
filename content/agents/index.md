---
title: Agents
tags:
  - agents
  - llm
  - workflows
---

# Agents

Agents는 LLM이 답변만 생성하는 것이 아니라, context를 모으고, tool을 쓰고, state를 추적하고, 결과를 검증해서 작업을 끝내는 방식입니다.

이 섹션은 세부 개념을 전부 sidebar에 펼치지 않고, 여섯 개 gateway로만 들어가게 정리합니다. 제품별 기능은 자주 바뀌므로, 먼저 오래 남는 구조와 workflow를 봅니다.

$$
\text{agent}
=
\text{model}
+ \text{context}
+ \text{tools}
+ \text{state}
+ \text{verification}
$$

## 먼저 볼 지도

| 질문 | 볼 곳 |
| --- | --- |
| agent가 내부적으로 어떻게 움직이는가? | [[agents/core/index|Core]] |
| 사용자가 실제로 누르는 기능은 무엇인가? | [[agents/features/index|Features]] |
| tool call, connector, side effect는 어떻게 다루는가? | [[agents/tools/index|Tools]] |
| coding, paper brief, LLM Wiki 같은 반복 작업은 어떻게 묶는가? | [[agents/workflows/index|Workflows]] |
| ChatGPT, Claude, Gemini, Copilot은 어떤 agent surface를 갖는가? | [[agents/models/index|Models]] |
| agent 결과를 어떻게 검증하고 완료 판단하는가? | [[agents/verification/index|Verification]] |

## 섹션

| 섹션 | 담는 것 | 대표 노트 |
| --- | --- | --- |
| [[agents/core/index|Core]] | architecture, loop, state, memory, planning | [[agents/core/agent-loop|Agent loop]], [[agents/core/context-engineering|Context engineering]] |
| [[agents/features/index|Features]] | chat, files, research, coding, memory, canvas, connectors | [[agents/features/files-and-knowledge|Files and knowledge]], [[agents/features/coding-workspace|Coding workspace]] |
| [[agents/tools/index|Tools]] | tool contract, permission, result handling | [[agents/tools/tool-use|Tool use]], [[agents/tools/tool-contract|Tool contract]] |
| [[agents/workflows/index|Workflows]] | 반복 가능한 작업 절차 | [[agents/workflows/coding-agents|Coding agents]], [[agents/workflows/llm-wiki|LLM Wiki]] |
| [[agents/models/index|Models]] | 공개 제품군의 agent interface | [[agents/models/chatgpt|ChatGPT]], [[agents/models/claude|Claude]] |
| [[agents/verification/index|Verification]] | acceptance criteria, evidence, audit, safety | [[agents/verification/verification-loop|Verification loop]], [[agents/verification/completion-audit|Completion audit]] |

## 분류 기준

| 내용 | 둘 곳 |
| --- | --- |
| prompt, context, memory, planning처럼 agent를 이루는 부품 | [[agents/core/index|Core]] |
| 사용자가 보는 기능 이름과 사용 패턴 | [[agents/features/index|Features]] |
| 외부 상태를 읽거나 바꾸는 인터페이스 | [[agents/tools/index|Tools]] |
| 여러 단계로 반복되는 작업 방식 | [[agents/workflows/index|Workflows]] |
| 제품별 차이와 interface 비교 | [[agents/models/index|Models]] |
| 결과가 맞는지 확인하는 기준과 절차 | [[agents/verification/index|Verification]] |

## 작성 규칙

- 개인 자동화보다 공개 제품에서 공통으로 보이는 pattern을 우선합니다.
- 세부 개념은 gateway 안에서 링크하고, sidebar에는 큰 묶음만 남깁니다.
- vendor-specific claim은 변동 가능성이 크므로 날짜, 범위, 공식 문서 확인 필요성을 남깁니다.
- private prompt, private repo detail, credential, internal task name, unpublished work는 공개하지 않습니다.
- “agent가 잘했다”보다 어떤 evidence로 확인했는지를 남깁니다.

## Related

- [[ai/index|AI]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]]
- [[agents/verification/completion-audit|Completion audit]]
