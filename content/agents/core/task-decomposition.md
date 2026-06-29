---
title: Task Decomposition
tags:
  - agents
  - planning
  - workflows
---

# Task Decomposition

Task decomposition은 broad goal을 agent가 execute하고 verify할 수 있는 bounded unit으로 나눕니다. Ambiguous user request와 concrete tool call 사이의 다리입니다.

Task는 subtask set으로 표현할 수 있습니다.

$$
T
=
\{t_1,t_2,\ldots,t_n\}
$$

Dependency는 아래처럼 둡니다.

$$
E
\subseteq
T \times T
$$

여기서 $(t_i,t_j)\in E$는 $t_j$가 $t_i$에 의존한다는 뜻입니다.

## 좋은 subtask

- clear artifact가 있습니다.
- explicit success check가 있습니다.
- 작은 context와 tool scope 안에 들어옵니다.
- discovery, editing, verification을 하나의 opaque step에 섞지 않습니다.
- failure를 diagnose할 만큼 local하게 만듭니다.

## Decomposition pattern

- Discover -> edit -> verify -> summarize.
- Draft -> review -> revise -> publish.
- Ingest -> sanitize -> link -> curate.
- Plan -> delegate -> admit patch -> verify.
- 큰 repetitive work는 folder, topic, risk level 기준으로 batch합니다.

## 확인할 것

- 각 subtask가 independently verifiable한가?
- action 전에 user approval이 필요한 subtask가 있는가?
- dependency가 explicit한가?
- conflict resolution owner가 하나로 정해져 있는가?
- correctness를 잃지 않고 plan을 줄일 수 있는가?

## Related

- [[agents/core/planning|Planning]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/workflows/agent-handoff|Agent handoff]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
