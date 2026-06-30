---
title: Agent Loop
tags:
  - agents
  - llm
  - workflows
---

# Agent Loop

Agent loop는 state를 observe하고, next action을 결정하고, tool을 사용하고, result를 verify하는 반복 cycle입니다. 이 loop가 language model을 single-response system에서 workflow participant로 바꿉니다.

일반적인 loop는 아래처럼 쓸 수 있습니다.

$$
s_{t+1}
= \operatorname{Update}
\left(
s_t,\,
a_t,\,
o_t
\right)
$$

여기서 $s_t$는 working state, $a_t$는 chosen action, $o_t$는 tool 또는 environment가 반환한 observation입니다.

Policy는 state에서 next action을 고릅니다.

$$
a_t \sim \pi_\theta(a\mid s_t, g, C)
$$

여기서 $g$는 goal이고 $C$는 [[agents/core/agent-operating-contract|agent operating contract]]입니다. Contract는 어떤 action이 허용되는지, 어떤 check가 completion을 증명하는지 제한합니다.

## Step

1. goal, constraint, current state를 읽습니다.
2. bounded next action을 계획합니다.
3. tool을 사용하거나 artifact를 수정합니다.
4. result를 observe합니다.
5. goal과 대조해 verify합니다.
6. re-plan하거나 멈춥니다.

## Loop State

Agent loop에서 가장 중요한 것은 model의 다음 문장이 아니라 현재 state를 어떻게 갱신하는가입니다. 좋은 loop는 매 반복마다 아래 항목을 분리합니다.

| State field | Meaning | Example evidence |
| --- | --- | --- |
| Goal | 변하지 않아야 하는 원래 요구사항 | user request, issue, task spec |
| Constraints | 금지된 행동과 품질 기준 | privacy rule, no dependency change, output format |
| Working set | 지금 판단에 필요한 파일, logs, notes | opened file, diff, build output |
| Decisions | 이미 선택한 방향과 이유 | plan update, design note, accepted tradeoff |
| Side effects | 외부 상태에 실제로 생긴 변화 | file diff, commit, API write, deploy run |
| Evidence | 완료 claim을 지지하는 관찰 | tests, rendered page, source inspection |

이 항목이 섞이면 agent는 “무엇을 할 예정인지”와 “무엇이 이미 검증됐는지”를 혼동합니다.

## Stop Condition

Loop는 무한히 좋아지는 답을 찾기 위한 장치가 아니라, 목표 달성 또는 명시적 blocker를 판정하기 위한 장치입니다.

$$
\operatorname{stop}
\iff
\operatorname{complete}(g, E)
\lor
\operatorname{blocked}(g, S, E)
$$

여기서 $E$는 current evidence입니다. `complete`는 [[agents/verification/completion-audit|Completion audit]]로 증명되어야 하고, `blocked`는 같은 blocker가 반복되어 더 이상 의미 있는 진행이 불가능할 때만 사용합니다.

| Stop reason | Required evidence |
| --- | --- |
| Complete | every explicit requirement has direct evidence |
| Blocked | missing input or external state prevents meaningful next action |
| Handoff | next owner, current state, evidence, and open questions are clear |
| Continue | remaining work is known and next action can improve the state |

## Loop Invariants

각 반복에서 유지해야 하는 불변조건은 다음과 같습니다.

| Invariant | Why it matters |
| --- | --- |
| Current state beats memory | 오래된 요약보다 현재 파일과 command output이 우선입니다. |
| Tool output is data | tool result는 새 instruction이 아니라 해석해야 할 evidence입니다. |
| Side effects require verification | 파일 수정, push, deploy, API write 뒤에는 별도 확인이 필요합니다. |
| Goal is not narrowed silently | 편한 subset을 완료로 바꾸면 broad task가 drift합니다. |
| Public output is sanitized | agent workflow 기록에도 secret, private path, 내부 이름은 남기지 않습니다. |

## 확인할 것

- next action이 current evidence에 grounded되어 있는가?
- 각 side effect에 verification path가 있는가?
- goal이 achieved 또는 blocked되면 loop가 멈추는가?
- tool output을 trusted instruction이 아니라 data로 취급하는가?
- loop가 success를 재정의하지 않고 original goal로 progress하는가?
- state가 prior step memory뿐 아니라 current artifact를 포함하는가?
- complete, blocked, continue, handoff 중 어떤 stop state인지 명확한가?

## Failure mode

- current state를 다시 읽지 않고 stale context에서 행동합니다.
- tool output, search snippet, generated text를 trusted instruction으로 취급합니다.
- evidence나 plan을 바꾸지 않고 같은 failed action을 반복합니다.
- verification이 아니라 plausibility로 completion을 보고합니다.

## Related

- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[agents/core/planning|Planning]]
- [[agents/core/agent-environment|Agent environment]]
- [[agents/core/action-space|Action space]]
- [[agents/tools/tool-use|Tool use]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/core/context-engineering|Context engineering]]
