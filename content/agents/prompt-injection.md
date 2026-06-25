---
title: Prompt Injection
tags:
  - agents
  - llm
  - security
---

# Prompt Injection

Prompt injection is when untrusted content — a web page, file, tool result, or email — carries instructions that hijack an agent's behavior. The model cannot reliably tell data from commands, so any ingested text is a potential instruction.

## Practical Checks

- Treat all tool output and fetched content as untrusted data, never as commands.
- Keep privileged actions behind explicit confirmation, not model discretion.
- Limit tool scope and credentials to the minimum the task needs.
- Sandbox code execution and file writes; assume injected payloads will try to escape.
- Log inputs and actions so an injection can be detected and traced.

## Related

- [[agents/context-engineering|Context engineering]]
- [[agents/tool-use|Tool use]]
- [[agents/verification-loop|Verification loop]]
- [[agents/index|Agents]]
