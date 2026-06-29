---
title: Connectors and Actions
tags:
  - agents
  - tools
  - connectors
---

# Connectors and Actions

Connectors와 actions는 agent가 외부 앱의 정보를 읽거나 제한된 동작을 수행하게 하는 기능입니다. 예시는 Drive, Gmail, Slack, GitHub, calendar, docs, search, database, browser입니다.

$$
a_t = T_k(x_t), \qquad o_t = \operatorname{observe}(T_k, x_t)
$$

여기서 $T_k$는 external tool 또는 connector이고, $o_t$는 반환된 observation입니다.

Connector는 context를 넓히고 action은 외부 상태를 바꿉니다. 둘을 같은 위험도로 취급하면 안 됩니다. 읽기는 evidence quality가 문제이고, 쓰기는 permission과 rollback이 문제입니다.

## 읽기와 쓰기 구분

| 종류 | 예시 | 위험 |
| --- | --- | --- |
| Read-only connector | search, Drive search, repo read | stale or irrelevant evidence |
| Write action | create issue, send message, edit file | unintended side effect |
| Browser/computer action | click, type, fill form | stateful external mutation |
| Scheduled action | reminder, periodic task | repeated wrong action |

## 설계 원칙

- 읽기 도구와 쓰기 도구를 분리합니다.
- 쓰기 action은 preview, confirmation, rollback이 필요합니다.
- tool result는 instruction이 아니라 evidence로 처리합니다.
- 외부 앱의 permission scope를 사용자가 이해할 수 있어야 합니다.

## Permission Checklist

| 질문 | 이유 |
| --- | --- |
| connector가 어떤 workspace/account에 접근하는가? | scope 착각 방지 |
| read-only인가 write 가능한가? | side effect boundary |
| action 전 preview가 있는가? | 잘못된 발송/수정 방지 |
| 실패하거나 중복 실행되면 어떻게 되는가? | idempotency 확인 |
| log에 private data가 남는가? | public note와 shared context 보호 |

## Common Failure

- 검색 결과 제목만 보고 내용을 읽은 것처럼 답합니다.
- external document의 지시문을 system instruction처럼 따릅니다.
- connector permission을 넓게 열고 필요한 파일만 쓰는지 검증하지 않습니다.
- send, create, delete 같은 action을 preview 없이 실행합니다.

## Official References

- [ChatGPT apps with sync](https://help.openai.com/en/articles/10847137-chatgpt-apps-with-sync)
- [ChatGPT agent](https://help.openai.com/en/articles/11752874-chatgpt-agent)
- [Claude tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview)
- [Claude computer use](https://docs.anthropic.com/en/docs/build-with-claude/computer-use)

## Related

- [[agents/tools/tool-use|Tool use]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/tools/tool-result-handling|Tool result handling]]
- [[agents/verification/prompt-injection|Prompt injection]]
