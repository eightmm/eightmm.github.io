---
title: Tool Use
tags:
  - agents
  - llm
  - tool-use
---

# Tool Use

Tool use는 LLM agent가 parameter만으로 답하는 대신 search, code execution, file edit, API call 같은 외부 function을 호출하게 하는 방식입니다. Model은 structured call을 만들고, harness가 실행한 뒤, 결과를 다시 context로 넣습니다.

기본 패턴은 아래와 같습니다.

$$
o_t = T_k(a_t; x_t)
$$

여기서 $T_k$는 tool $k$, $x_t$는 typed argument payload, $o_t$는 agent에게 돌아오는 observation입니다.

Tool use는 확실성으로 가는 shortcut이 아니라 environment와의 interaction으로 봐야 합니다. Tool output은 partial, stale, adversarial일 수 있고, 현재 claim에 비해 너무 넓을 수도 있습니다.

## Tool category

- Read-only tool: search, file read, log, status query.
- Editing tool: file patch, formatting, migration.
- Execution tool: test, build, script, notebook.
- External side-effect tool: API write, deployment, email, issue update.
- Review tool: diff inspection, static analysis, human review, multi-agent review.

## 실전 check

- 각 tool은 명확한 이름, typed argument, 사용할 때를 설명하는 짧은 description을 가져야 합니다.
- 실행 전에 argument를 validate합니다. Model output을 safe input으로 그대로 믿지 않습니다.
- 결과는 간결하고 structured하게 반환합니다. 큰 dump는 context를 낭비하고 signal을 숨깁니다.
- side effect가 있는 tool은 idempotent하게 만들거나 confirmation으로 gate합니다.
- run을 replay하고 audit할 수 있도록 모든 call을 log합니다.
- 필요한 evidence를 줄 수 있는 가장 좁은 tool을 우선합니다.
- 큰 output은 command와 failure line을 보존하면서 actionable evidence로 요약합니다.

## Failure mode

- source를 열지 않고 broad search result를 proof로 쓰는 경우.
- tool result가 agent policy에 새 instruction을 주입하게 두는 경우.
- 명시적 boundary 없이 destructive action을 실행하는 경우.
- 실패한 command를 generic success summary 뒤에 숨기는 경우.

## Related

- [[concepts/llm/tool-calling|Tool calling]]
- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/tools/tool-result-handling|Tool result handling]]
- [[agents/core/planning|Planning]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/prompt-injection|Prompt injection]]
- [[agents/index|Agents]]
