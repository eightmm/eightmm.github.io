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

Orchestration은 agent 수를 늘리는 기술이 아니라, artifact flow와 responsibility를 분리하는 기술입니다. 단일 agent loop로 충분한 작업에 multi-agent를 붙이면 coordination cost만 늘어납니다.

## Pattern

- Pipeline: 한 step이 다음 step의 input을 만듭니다.
- Supervisor-worker: 하나의 controller가 bounded task를 delegate합니다.
- Reviewer loop: second pass가 first artifact를 check합니다.
- Debate or council: independent agent들이 synthesis 전에 의견을 냅니다.
- Human gate: human이 high-risk transition을 승인합니다.

## Artifact Flow

Orchestration의 핵심은 agent 수가 아니라 artifact contract입니다.

$$
a_{t+1} = F_i(a_t, c_i, e_i)
$$

여기서 $a_t$는 artifact, $c_i$는 step context, $e_i$는 evidence 또는 tool result입니다.

| Artifact | Producer | Consumer | Check |
| --- | --- | --- | --- |
| candidate list | discovery agent | curator | source and scope |
| draft note | writer agent | editor | links and uncertainty |
| diff or patch | coding agent | reviewer | scope and tests |
| review finding | reviewer | owner | reproducibility |
| verification log | verifier | publisher | commands and outputs |

If the artifact cannot be inspected, orchestration becomes conversation, not workflow.

## Control Plane

The controller should decide:

| Decision | Why |
| --- | --- |
| task split | prevents agents from duplicating work |
| context budget | limits irrelevant history and prompt contamination |
| write permission | prevents uncontrolled repository edits |
| acceptance criteria | tells reviewers what evidence matters |
| conflict resolution | avoids merging incompatible outputs |

For public wiki work, the owner should prefer small deterministic handoffs: one note, one route, one verifier.

## When to Use

| 상황 | orchestration이 유용한 이유 |
| --- | --- |
| 많은 후보를 수집해야 함 | parallel discovery가 시간 단축 |
| 중요한 변경을 검토해야 함 | independent review로 blind spot 감소 |
| 서로 다른 전문성이 필요함 | coding, research, security, writing 역할 분리 |
| 긴 workflow를 재사용해야 함 | runbook과 handoff artifact로 반복 가능 |

## When Not to Use

- task가 작고 context가 하나의 agent에 충분히 들어갑니다.
- shared state가 불명확해서 agent들이 서로 덮어쓸 위험이 큽니다.
- verifier 없이 의견만 여러 개 모읍니다.
- delegation 결과를 admit하거나 review할 owner가 없습니다.

## 확인할 것

- step 사이에 어떤 artifact가 전달되는가?
- agent들이 independent한가, 아니면 같은 context bias를 공유하는가?
- conflict는 누가 해결하는가?
- 다른 agent의 confidence를 믿지 않고 각 step을 verify할 수 있는가?
- orchestration이 single well-scoped agent loop보다 실제 value를 더하는가?
- delegated output을 current repository나 source state에 다시 맞춰 확인하는가?
- artifact가 inspect 가능한 파일, diff, table, command output 중 하나인가?
- human gate가 필요한 공개/보안/연구 claim 전환이 있는가?

## Related

- [[agents/workflows/agent-handoff|Agent handoff]]
- [[agents/workflows/agent-runbook|Agent runbook]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[agents/verification/verification-loop|Verification loop]]
- [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]]
