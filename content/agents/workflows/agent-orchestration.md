---
title: Agent Orchestration
tags:
  - agents
  - workflows
  - multi-agent
---

# Agent Orchestration

Agent orchestration은 여러 model call, tool, role, agent를 하나의 workflow로 조율합니다. Task가 discovery, drafting, implementation, review, verification으로 자연스럽게 나뉠 때 유용합니다.

Workflow는 directed graph로 표현할 수 있습니다.

$$
G = (V, E)
$$

여기서 node $V$는 step 또는 role이고, edge $E$는 artifact, evidence, decision을 전달합니다.

## Pattern

- Pipeline: 한 step이 다음 step의 input을 만듭니다.
- Supervisor-worker: 하나의 controller가 bounded task를 delegate합니다.
- Reviewer loop: second pass가 first artifact를 check합니다.
- Debate or council: independent agent들이 synthesis 전에 의견을 냅니다.
- Human gate: human이 high-risk transition을 승인합니다.

## 확인할 것

- step 사이에 어떤 artifact가 전달되는가?
- agent들이 independent한가, 아니면 같은 context bias를 공유하는가?
- conflict는 누가 해결하는가?
- 다른 agent의 confidence를 믿지 않고 각 step을 verify할 수 있는가?
- orchestration이 single well-scoped agent loop보다 실제 value를 더하는가?

## Related

- [[agents/workflows/agent-handoff|Agent handoff]]
- [[agents/workflows/agent-runbook|Agent runbook]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[agents/verification/verification-loop|Verification loop]]
- [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]]
