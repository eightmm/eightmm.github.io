---
title: Agent State
tags:
  - agents
  - workflows
  - memory
---

# Agent State

Agent state는 agent가 다음에 무엇을 할지 결정할 때 쓰는 structured information입니다. Conversation에 우연히 남아 있는 모든 것을 state로 취급하면 안 됩니다.

유용한 state는 아래처럼 모델링할 수 있습니다.

$$
s_t = (g, C, E_t, P_t, A_t)
$$

여기서 $g$는 goal, $C$는 constraint, $E_t$는 accumulated evidence, $P_t$는 plan, $A_t$는 action history입니다.

## State type

- Goal state: objective, scope, success criteria, stop condition.
- Environment state: file, command output, browser state, repository status, external artifact.
- Evidence state: tool 또는 primary source에서 모은 fact.
- Plan state: pending, active, completed step.
- Risk state: approval, destructive action, secret, public/private boundary.

## 확인할 것

- goal이 여전히 newest user request인가?
- current plan이 evidence에 기반하는가, stale memory에 기반하는가?
- assumption과 observed fact가 분리되어 있는가?
- private detail이 durable public note에서 제외되는가?
- context change 뒤에도 state가 misleading하지 않게 유지되는가?

## Related

- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/agent-memory|Agent memory]]
- [[agents/core/planning|Planning]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
