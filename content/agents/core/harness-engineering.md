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

## 좋은 harness의 특징

- tool call과 natural-language answer를 구분합니다.
- 읽기 action과 쓰기 action의 권한을 다르게 둡니다.
- 실패한 tool result를 숨기지 않습니다.
- final answer가 어떤 evidence에 의존하는지 남깁니다.
- context overflow 때 오래된 추정을 사실처럼 유지하지 않습니다.

## Related

- [[agents/core/agent-loop|Agent loop]]
- [[agents/core/agent-state|Agent state]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/verification/verification-loop|Verification loop]]
