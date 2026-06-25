---
title: Tool Use
tags:
  - agents
  - llm
  - tool-use
---

# Tool Use

Tool use lets an LLM agent call external functions — search, code execution, file edits, API calls — instead of answering from parameters alone. The model emits a structured call, the harness runs it, and the result is fed back into context.

## Practical Checks

- Define each tool with a clear name, typed arguments, and a short description of when to use it.
- Validate arguments before executing; never trust model output as safe input.
- Return concise, structured results — large dumps waste context and hide signal.
- Make side-effecting tools idempotent or gated by confirmation.
- Log every call so a run can be replayed and audited.

## Related

- [[concepts/llm/tool-calling|Tool calling]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/core/planning|Planning]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/prompt-injection|Prompt injection]]
- [[agents/index|Agents]]
