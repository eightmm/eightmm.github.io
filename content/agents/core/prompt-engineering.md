---
title: Prompt Engineering
tags:
  - agents
  - prompting
---

# Prompt Engineering

Prompt engineering은 모델에게 예쁜 문장을 쓰게 하는 기술이 아니라, 작업의 목적과 제약을 context 안에서 실행 가능한 형태로 만드는 기술입니다.

$$
p^\ast = \arg\max_p \mathbb{E}[U(y, g) \mid p, C, M]
$$

여기서 $p$는 prompt, $C$는 context, $M$은 model behavior, $U$는 task utility입니다.

Prompt는 model weight를 바꾸지 않습니다. 대신 현재 call에서 task, evidence, constraint, allowed action, output contract를 정렬합니다. 그래서 agent prompt는 “말투”보다 “무엇을 근거로 어떤 행동을 해도 되는가”를 더 명확히 해야 합니다.

## 기본 구조

| Field | 내용 |
| --- | --- |
| Task | 무엇을 해야 하는가 |
| Context | 어떤 정보를 보고 판단해야 하는가 |
| Constraints | 금지, 형식, 범위, 품질 기준 |
| Process | 필요한 경우 단계, 체크, 비교 기준 |
| Output | 표, bullet, patch, report, decision 등 |

## Prompt Layer

| Layer | 역할 | 흔한 실패 |
| --- | --- | --- |
| Role | model이 맡은 작업 태도와 전문성 | role만 있고 success criteria가 없음 |
| Task | 이번 turn의 concrete objective | broad aspiration으로 남음 |
| Context | 판단에 필요한 evidence | stale summary와 current state가 섞임 |
| Constraint | 금지, 범위, public/private boundary | destructive action이나 secret 처리 기준 누락 |
| Tool policy | 어떤 tool을 언제 써야 하는지 | tool output을 instruction으로 오해 |
| Output contract | 사용자가 받을 artifact 형태 | verified fact와 assumption이 섞임 |

## Agent prompt에서 더 중요한 것

Agent prompt는 답변 방식뿐 아니라 행동 경계를 정합니다.

- 어떤 도구를 사용할 수 있는가.
- 어떤 작업은 사용자 확인이 필요한가.
- tool result를 instruction이 아니라 evidence로 볼 것인가.
- 언제 멈추고 완료를 주장할 수 있는가.

## Good Prompt Checks

- task success가 observable artifact로 정의되어 있는가?
- 최신 user request가 오래된 instruction보다 우선된다는 점이 분명한가?
- public note에 넣으면 안 되는 private detail이 명시되어 있는가?
- model이 모르면 어떤 source를 inspect해야 하는지 적혀 있는가?
- output이 review 가능한 형태로 제한되어 있는가?
- verification 없이 completion을 말하지 않도록 되어 있는가?

## Related

- [[concepts/llm/prompting|Prompting]]
- [[agents/features/chat-and-prompting|Chat and prompting]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/agent-operating-contract|Agent operating contract]]
