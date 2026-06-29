---
title: Agent Environment
tags:
  - agents
  - environment
---

# Agent Environment

Agent environment는 agent가 observe하거나 change할 수 있는 external state입니다. File, repository, terminal, web page, API, queue, issue tracker, generated artifact가 포함될 수 있습니다.

간단한 transition view는 아래와 같습니다.

$$
s_{t+1} = T(s_t, a_t, e_t)
$$

$s_t$는 agent의 working state, $a_t$는 selected action, $e_t$는 environment state, $T$는 action과 observation이 만든 transition입니다.

## Environment type

- Read-only environment: search result, log, documentation, repository file.
- Editable environment: source file, Markdown note, configuration, generated artifact.
- External environment: API, CI system, deployment target, issue tracker.
- Human environment: approval, feedback, task constraint, review decision.

## 중요한 이유

Agent는 text context와 actual external state를 혼동할 때 실패합니다. Conversation에 쓴 plan은 file이 바뀌었거나 build가 통과했거나 deployment가 성공했다는 증거가 아닙니다. Environment는 직접 inspect해야 합니다.

## 확인할 것

- agent가 어떤 external state를 observe할 수 있는가?
- agent가 어떤 external state를 change할 수 있는가?
- 어떤 action이 read-only이고 어떤 action이 side-effecting인가?
- environment가 의도대로 바뀌었음을 어떤 evidence가 증명하는가?
- generated 또는 out of scope라서 무시해야 할 state는 무엇인가?

## Related

- [[agents/core/agent-loop|Agent loop]]
- [[agents/core/agent-state|Agent state]]
- [[agents/core/action-space|Action space]]
- [[agents/tools/tool-use|Tool use]]
- [[agents/verification/verification-loop|Verification loop]]
