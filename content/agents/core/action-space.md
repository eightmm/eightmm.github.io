---
title: Action Space
tags:
  - agents
  - planning
---

# Action Space

Action space는 특정 step에서 agent가 선택할 수 있는 action의 집합입니다. 질문하기 같은 natural-language action과 file 읽기, note 수정, command 실행, pull request 열기 같은 tool action이 모두 포함됩니다.

Formally:

$$
a_t \in \mathcal{A}(s_t, b_t)
$$

$a_t$는 step $t$의 action, $\mathcal{A}$는 available action set, $s_t$는 task state, $b_t$는 permission 또는 boundary condition입니다.

## Action class

- Observe: file, log, page, command output을 inspect합니다.
- Transform: source를 edit하고, note를 rewrite하고, evidence를 summarize하고, text를 refactor합니다.
- Execute: build, test, formatter, deployment, script를 실행합니다.
- Ask: missing constraint, approval, domain judgment를 요청합니다.
- Stop: completion, failure, blocked state를 evidence와 함께 보고합니다.

## Constraint

모든 possible action이 모든 step에서 available하면 안 됩니다. Side-effecting action에는 clear purpose와 verification path가 필요합니다. Destructive, expensive, private, externally visible action에는 더 강한 boundary가 필요합니다.

## 확인할 것

- next action이 allowed action space 안에 있는가?
- editing 전에 read-only action만으로 충분한가?
- action이 bounded result를 갖는가?
- 어떤 verifier가 action result를 inspect할 것인가?
- action 전에 agent가 물어야 하는가?

## Related

- [[agents/core/planning|Planning]]
- [[agents/core/agent-environment|Agent environment]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[agents/verification/verification-loop|Verification loop]]
