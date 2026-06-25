---
title: Tool Use
tags:
  - agents
  - llm
  - tool-use
---

# Tool Use

Tool use lets an LLM agent call external functions — search, code execution, file edits, API calls — instead of answering from parameters alone. The model emits a structured call, the harness runs it, and the result is fed back into context.

The basic pattern is:

$$
o_t = T_k(a_t; x_t)
$$

where $T_k$ is tool $k$, $x_t$ is the typed argument payload, and $o_t$ is the observation returned to the agent.

Tool use should be treated as interaction with an environment, not as a shortcut to certainty. Tool outputs can be partial, stale, adversarial, or too broad for the current claim.

## Tool Categories

- Read-only tools: search, file reads, logs, status queries.
- Editing tools: file patches, formatting, migrations.
- Execution tools: tests, builds, scripts, notebooks.
- External side-effect tools: API writes, deployment, email, issue updates.
- Review tools: diff inspection, static analysis, human review, multi-agent review.

## Practical Checks

- Define each tool with a clear name, typed arguments, and a short description of when to use it.
- Validate arguments before executing; never trust model output as safe input.
- Return concise, structured results — large dumps waste context and hide signal.
- Make side-effecting tools idempotent or gated by confirmation.
- Log every call so a run can be replayed and audited.
- Prefer the narrowest tool that can provide the needed evidence.
- Summarize large outputs into actionable evidence while preserving commands and failure lines.

## Failure Modes

- Using a broad search result as proof without opening the source.
- Letting a tool result inject new instructions into the agent's policy.
- Running destructive actions without an explicit boundary.
- Hiding failed commands behind a generic success summary.

## Related

- [[concepts/llm/tool-calling|Tool calling]]
- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/tools/tool-result-handling|Tool result handling]]
- [[agents/core/planning|Planning]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/prompt-injection|Prompt injection]]
- [[agents/index|Agents]]
