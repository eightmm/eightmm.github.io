---
title: Agents
tags:
  - agents
  - llm
  - workflows
---

# Agents

Agents는 LLM이 답변만 생성하는 것이 아니라, context를 모으고, tool을 쓰고, state를 추적하고, 결과를 검증해서 작업을 끝내는 방식입니다.

이 섹션은 제품 기능 목록보다 오래 남는 agent의 원리를 먼저 봅니다. 핵심은 model, context, tools, state, verification이 어떻게 묶여 하나의 작업 흐름을 만드는지입니다.

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
| agent가 내부적으로 어떻게 움직이는가? | [Core](/agents/core) |
| 사용자가 실제로 누르는 기능은 무엇인가? | [Features](/agents/features) |
| tool call, connector, side effect는 어떻게 다루는가? | [Tools](/agents/tools) |
| coding, paper brief, LLM Wiki 같은 반복 작업은 어떻게 묶는가? | [Workflows](/agents/workflows) |
| ChatGPT, Claude, Gemini, Copilot은 어떤 agent surface를 갖는가? | [Models](/agents/models) |
| agent 결과를 어떻게 검증하고 완료 판단하는가? | [Verification](/agents/verification) |

## 섹션

| 섹션 | 담는 것 | 대표 노트 |
| --- | --- | --- |
| [Core](/agents/core) | architecture, loop, state, memory, planning | [Agent loop](/agents/core/agent-loop), [Context engineering](/agents/core/context-engineering) |
| [Features](/agents/features) | chat, files, research, coding, memory, canvas, connectors | [Files and knowledge](/agents/features/files-and-knowledge), [Coding workspace](/agents/features/coding-workspace) |
| [Tools](/agents/tools) | tool contract, permission, result handling | [Tool use](/agents/tools/tool-use), [Tool contract](/agents/tools/tool-contract) |
| [Workflows](/agents/workflows) | 반복 가능한 작업 절차 | [Coding agents](/agents/workflows/coding-agents), [LLM Wiki](/agents/workflows/llm-wiki) |
| [Models](/agents/models) | 공개 제품군의 agent interface | [ChatGPT](/agents/models/chatgpt), [Claude](/agents/models/claude) |
| [Verification](/agents/verification) | acceptance criteria, evidence, audit, safety | [Verification loop](/agents/verification/verification-loop), [Completion audit](/agents/verification/completion-audit) |

## 분류 기준

| 내용 | 둘 곳 |
| --- | --- |
| prompt, context, memory, planning처럼 agent를 이루는 부품 | [Core](/agents/core) |
| 사용자가 보는 기능 이름과 사용 패턴 | [Features](/agents/features) |
| 외부 상태를 읽거나 바꾸는 인터페이스 | [Tools](/agents/tools) |
| 여러 단계로 반복되는 작업 방식 | [Workflows](/agents/workflows) |
| 제품별 차이와 interface 비교 | [Models](/agents/models) |
| 결과가 맞는지 확인하는 기준과 절차 | [Verification](/agents/verification) |

## 작성 규칙

- 개인 자동화보다 공개 제품에서 공통으로 보이는 pattern을 우선합니다.
- 세부 개념은 Core, Tools, Workflows, Verification 같은 안정적인 묶음 안에서 연결합니다.
- vendor-specific claim은 변동 가능성이 크므로 날짜, 범위, 공식 문서 확인 필요성을 남깁니다.
- private prompt, private repo detail, credential, internal task name, unpublished work는 공개하지 않습니다.
- “agent가 잘했다”보다 어떤 evidence로 확인했는지를 남깁니다.

## Related

- [[ai/index|AI]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]]
- [[agents/verification/completion-audit|Completion audit]]
