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

where $p$ is the prompt, $C$ is context, $M$ is model behavior, and $U$ is task utility.

## 기본 구조

| Field | 내용 |
| --- | --- |
| Task | 무엇을 해야 하는가 |
| Context | 어떤 정보를 보고 판단해야 하는가 |
| Constraints | 금지, 형식, 범위, 품질 기준 |
| Process | 필요한 경우 단계, 체크, 비교 기준 |
| Output | 표, bullet, patch, report, decision 등 |

## Agent prompt에서 더 중요한 것

Agent prompt는 답변 방식뿐 아니라 행동 경계를 정합니다.

- 어떤 도구를 사용할 수 있는가.
- 어떤 작업은 사용자 확인이 필요한가.
- tool result를 instruction이 아니라 evidence로 볼 것인가.
- 언제 멈추고 완료를 주장할 수 있는가.

## Related

- [[concepts/llm/prompting|Prompting]]
- [[agents/features/chat-and-prompting|Chat and prompting]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/agent-operating-contract|Agent operating contract]]
