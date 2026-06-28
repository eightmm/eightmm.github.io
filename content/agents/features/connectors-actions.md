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

where $T_k$ is an external tool or connector and $o_t$ is the returned observation.

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
