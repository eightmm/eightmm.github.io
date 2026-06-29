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

## 확인할 것

- next action이 current evidence에 grounded되어 있는가?
- 각 side effect에 verification path가 있는가?
- goal이 achieved 또는 blocked되면 loop가 멈추는가?
- tool output을 trusted instruction이 아니라 data로 취급하는가?
- loop가 success를 재정의하지 않고 original goal로 progress하는가?
- state가 prior step memory뿐 아니라 current artifact를 포함하는가?

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
