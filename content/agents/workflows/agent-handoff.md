---
title: Agent Handoff
tags:
  - agents
  - workflows
  - handoff
---

# Agent Handoff

Agent handoff는 task, artifact, decision을 한 agent 또는 role에서 다른 쪽으로 넘기는 과정입니다. 좋은 handoff는 다음 agent가 hidden context를 믿지 않아도 바로 생산적으로 일하게 만듭니다.

Handoff에는 아래가 포함되어야 합니다.

$$
H
=
\{\text{goal},\text{scope},\text{state},\text{artifacts},\text{evidence},\text{risks},\text{next step}\}
$$

## Handoff 내용

- Goal: 아직 해야 하는 일.
- Scope: 범위 안에 있는 file, topic, system.
- Current state: 무엇이 바뀌었고 무엇이 남았는지.
- Artifacts: path, commit, output, draft.
- Evidence: 이미 실행한 check와 결과.
- Risks: 알려진 uncertainty, failed check, sensitive boundary.
- Next step: 가장 작은 useful continuation.

## 흔한 handoff 유형

- Research handoff: paper candidate와 verification status.
- Coding handoff: patch summary, test, risk area.
- Review handoff: finding, severity, evidence.
- Operations handoff: failure record, last good state, recovery plan.
- Wiki handoff: new page, broken link, stub, curation queue.

## 확인할 것

- receiver가 claimed state를 verify할 수 있는가?
- private detail이 제거되었거나 일반화되었는가?
- completed step과 pending step이 분리되어 있는가?
- skipped check가 명확히 표시되어 있는가?
- next step이 실행할 만큼 구체적인가?

## Related

- [[agents/workflows/agent-orchestration|Agent orchestration]]
- [[agents/core/task-decomposition|Task decomposition]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[concepts/systems/failure-recovery|Failure recovery]]
