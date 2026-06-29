---
title: Harness Engineering
tags:
  - agents
  - harness
  - engineering
---

# Harness Engineering

Harness는 모델 호출을 실제 workflow로 감싸는 실행층입니다. 사용자는 보통 harness를 직접 보지 않지만, agent의 품질은 harness가 context, tool, state, permission, verification을 어떻게 다루는지에 크게 의존합니다.

$$
\text{harness}
=
\{\text{state manager},\text{context builder},\text{tool runner},\text{policy gate},\text{verifier}\}
$$

## Harness가 하는 일

| Component | 역할 |
| --- | --- |
| State manager | goal, plan, observations, artifacts 추적 |
| Context builder | 현재 step에 필요한 정보만 prompt로 구성 |
| Tool runner | tool call schema 검증, 실행, 결과 반환 |
| Policy gate | 위험 행동, 권한, confirmation 처리 |
| Verifier | 테스트, citation, diff, acceptance check 실행 |

## Harness Boundary

Harness는 model architecture가 아닙니다. 같은 model이라도 harness가 다르면 agent behavior가 달라집니다. 특히 coding agent, browsing agent, paper brief agent는 필요한 tool runner와 verifier가 서로 다릅니다.

| Workflow | Harness가 강하게 다뤄야 하는 것 |
| --- | --- |
| Coding | git state, patch scope, test command, diff summary |
| Research browsing | source freshness, citation, conflicting evidence |
| LLM Wiki | note routing, wikilink, public/private sanitization |
| Multi-agent review | independent context, patch admission, conflict resolution |

## 좋은 harness의 특징

- tool call과 natural-language answer를 구분합니다.
- 읽기 action과 쓰기 action의 권한을 다르게 둡니다.
- 실패한 tool result를 숨기지 않습니다.
- final answer가 어떤 evidence에 의존하는지 남깁니다.
- context overflow 때 오래된 추정을 사실처럼 유지하지 않습니다.

## Failure Mode

- tool failure를 숨기고 natural-language fallback으로 성공처럼 답합니다.
- context packing이 오래된 summary를 최신 source보다 우선합니다.
- side-effecting tool을 read-only tool처럼 다룹니다.
- verifier가 optional이 되어 completion claim을 막지 못합니다.
- multiple agent output을 합치지만 conflict와 evidence quality를 기록하지 않습니다.

## Related

- [[agents/core/agent-loop|Agent loop]]
- [[agents/core/agent-state|Agent state]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/verification/verification-loop|Verification loop]]
