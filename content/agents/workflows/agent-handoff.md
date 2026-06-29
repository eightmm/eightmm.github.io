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

Handoff의 목적은 대화를 압축하는 것이 아니라, 다음 작업자가 current state를 재구성할 수 있게 하는 것입니다. 따라서 “내가 뭘 했는지”보다 “무엇을 증명했고 무엇이 아직 불확실한지”가 더 중요합니다.

## Handoff 내용

- Goal: 아직 해야 하는 일.
- Scope: 범위 안에 있는 file, topic, system.
- Current state: 무엇이 바뀌었고 무엇이 남았는지.
- Artifacts: path, commit, output, draft.
- Evidence: 이미 실행한 check와 결과.
- Risks: 알려진 uncertainty, failed check, sensitive boundary.
- Next step: 가장 작은 useful continuation.

## Handoff Template

| Field | 적을 내용 |
| --- | --- |
| Objective | 원래 목표와 최신 요청 |
| Current state | authoritative file, branch, URL, run 상태 |
| Changed | 수정된 artifact와 이유 |
| Verified | 실행한 check와 결과 |
| Not verified | 생략되었거나 약한 check |
| Dirty/excluded | 의도적으로 건드리지 않은 변경 |
| Next | 이어서 할 수 있는 가장 작은 작업 |

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
- handoff가 오래된 memory보다 current file/output을 우선하게 만드는가?

## Related

- [[agents/workflows/agent-orchestration|Agent orchestration]]
- [[agents/core/task-decomposition|Task decomposition]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[concepts/systems/failure-recovery|Failure recovery]]
