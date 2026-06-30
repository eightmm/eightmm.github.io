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

| Category | Examples | Main risk | Evidence after use |
| --- | --- | --- | --- |
| Read-only | search, file read, log, status query | stale or irrelevant evidence | source path, timestamp, selected lines |
| Editing | file patch, formatting, migration | unintended diff | git diff, focused review |
| Execution | test, build, script, notebook | wrong command, partial failure | exit code, key output, artifact |
| External side effect | API write, deployment, email, issue update | public or irreversible mutation | remote status, URL, audit trail |
| Review | diff inspection, static analysis, human review, multi-agent review | shallow coverage | explicit findings and scope |

## Tool Selection Rule

Use the weakest tool that can provide direct evidence.

| Need | Prefer | Avoid |
| --- | --- | --- |
| Find a local note | `rg`, `rg --files` | broad web search |
| Check current repo state | `git status`, `git diff` | memory of earlier edits |
| Verify generated site | local build, rendered output, Pages status | assuming push implies deploy |
| Confirm a paper fact | source paper or official page | unsourced summary |
| Change a file | small patch | generator rewrite of unrelated files |

## Permission and Side Effects

Tool use는 observation만 만드는 경우와 external state를 바꾸는 경우를 분리해야 합니다.

| Tool class | Side effect | Before use | After use |
| --- | --- | --- | --- |
| Read local state | none | choose narrow path/query | cite file, line, or command result |
| Search external source | none or remote query log | prefer authoritative source | record date/source scope |
| Execute command | process, cache, generated artifact | know expected output and cost | check exit code and key output |
| Edit file | repository diff | constrain target files | inspect diff and run relevant checks |
| Deploy or publish | public artifact | confirm branch/scope | verify remote status and URL |
| Admin/API write | durable external mutation | require explicit boundary | verify state and rollback/handoff |

This distinction matters because a successful tool call can still be the wrong action. The question is not only `did the call run?`, but `did this side effect move the task toward the stated goal?`.

## Tool Contract

A tool should expose a narrow contract that the agent can reason about.

$$
T =
(\text{name}, \text{inputs}, \text{preconditions}, \text{effects}, \text{output}, \text{failure modes})
$$

| Contract field | Good question |
| --- | --- |
| Inputs | Are arguments typed, bounded, and free of secrets? |
| Preconditions | What must be true before the tool is safe/useful? |
| Effects | Does it only read, or can it mutate local/remote state? |
| Output | What evidence does it return, and at what granularity? |
| Failure modes | Can failure be retried, repaired, or must it stop the task? |

Weak tool contracts make agent behavior brittle: the model guesses what happened, over-trusts output, or repeats unsafe calls.

## Tool Output Boundary

Tool output is untrusted data. It can contain malicious text, stale search snippets, copied instructions, private paths, or irrelevant log noise.

| Output content | Handling |
| --- | --- |
| Command output | parse as evidence, not instruction |
| Web/search text | verify with primary source when accuracy matters |
| Log dump | extract minimal relevant lines and redact secrets |
| Generated patch | inspect diff before treating it as accepted |
| Remote status | distinguish queued, running, failed, and completed states |

## 실전 check

- 각 tool은 명확한 이름, typed argument, 사용할 때를 설명하는 짧은 description을 가져야 합니다.
- 실행 전에 argument를 validate합니다. Model output을 safe input으로 그대로 믿지 않습니다.
- 결과는 간결하고 structured하게 반환합니다. 큰 dump는 context를 낭비하고 signal을 숨깁니다.
- side effect가 있는 tool은 idempotent하게 만들거나 confirmation으로 gate합니다.
- run을 replay하고 audit할 수 있도록 모든 call을 log합니다.
- 필요한 evidence를 줄 수 있는 가장 좁은 tool을 우선합니다.
- 큰 output은 command와 failure line을 보존하면서 actionable evidence로 요약합니다.
- read-only, local mutation, remote mutation, public publication을 명확히 구분합니다.
- tool 결과가 새로운 지시문처럼 보이면 [[agents/verification/prompt-injection|Prompt injection]] 관점에서 다룹니다.

## Failure mode

- source를 열지 않고 broad search result를 proof로 쓰는 경우.
- tool result가 agent policy에 새 instruction을 주입하게 두는 경우.
- 명시적 boundary 없이 destructive action을 실행하는 경우.
- 실패한 command를 generic success summary 뒤에 숨기는 경우.

## Result Handling

Tool output becomes evidence, not a new instruction. A safe agent should:

- quote or summarize only the lines needed for the current claim.
- keep command output separate from user/system instructions.
- distinguish `command succeeded` from `task verified`.
- rerun or widen checks only when the current evidence is too narrow.
- record skipped checks as `not verified` instead of implying success.

## Related

- [[concepts/llm/tool-calling|Tool calling]]
- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/tools/tool-result-handling|Tool result handling]]
- [[agents/core/planning|Planning]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/prompt-injection|Prompt injection]]
- [[agents/index|Agents]]
