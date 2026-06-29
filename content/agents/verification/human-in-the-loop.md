---
title: Human in the Loop
tags:
  - agents
  - workflows
  - review
---

# Human in the Loop

Human-in-the-loop design은 중요한 decision은 human reviewer에게 남기고, search, drafting, editing, routine verification은 agent에게 위임하는 방식입니다.

핵심 분리는 아래와 같습니다.

$$
\operatorname{Decision}
=
\operatorname{HumanReview}
\left(
\operatorname{AgentProposal},
\operatorname{Evidence}
\right)
$$

Agent는 proposal과 evidence를 inspect하기 쉽게 만들어야 합니다. Ambiguous하거나 high-risk한 choice에 대한 judgment는 human이 책임집니다.

Human-in-the-loop은 agent를 약하게 만드는 장치가 아니라, 책임과 판단을 올바른 위치에 두는 interface입니다. Agent는 반복 작업과 evidence collection을 맡고, human은 value judgment, risk acceptance, publication decision을 맡습니다.

## Review Boundary

| Human이 봐야 하는 것 | 이유 |
| --- | --- |
| Public publication | private detail, unpublished claim, tone, audience risk |
| Security-sensitive change | credential, permission, network exposure risk |
| Expensive execution | GPU/Slurm/cloud cost와 queue impact |
| Research claim | evidence strength, novelty, collaborator boundary |
| Destructive edit | rollback cost와 data loss risk |

## 좋은 handoff

- agent가 무엇이 바뀌었고 어떻게 verify했는지 적습니다.
- agent가 fact, assumption, unresolved question을 분리합니다.
- human이 data, security, dependency, public claim 같은 high-risk surface를 review합니다.
- final artifact가 decision을 나중에 다시 볼 수 있을 만큼 provenance를 기록합니다.

## 확인할 것

- 어떤 decision에 explicit human approval이 필요한가?
- full agent transcript를 읽지 않아도 artifact를 review할 수 있는가?
- uncertain claim을 흐리지 않고 표시하는가?
- publication 전에 public note를 sanitize하는가?

## Bad Pattern

- human에게 full transcript를 읽게 하고 실제 decision summary를 주지 않습니다.
- agent confidence를 evidence처럼 취급합니다.
- “나중에 확인”을 남긴 채 public artifact를 publish합니다.
- approval이 필요한 action을 작은 작업처럼 보이게 쪼갭니다.

## Related

- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[papers/workflows/paper-note-format|Paper note format]]
