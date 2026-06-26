---
title: Prompt Injection
tags:
  - agents
  - llm
  - security
---

# Prompt Injection

Prompt injection is when untrusted content — a web page, file, tool result, or email — carries instructions that hijack an agent's behavior. The model cannot reliably tell data from commands, so any ingested text is a potential instruction.

The central issue is a broken trust boundary:

$$
\text{untrusted data}
\not\Rightarrow
\text{trusted instruction}
$$

If an agent mixes retrieved data, tool output, user instructions, and system policy into one context, the model may follow text that should have been treated as evidence only.

## Common Pattern

A prompt-injection payload often tries to:

1. Override previous instructions.
2. Exfiltrate secrets or private context.
3. Trigger a privileged tool call.
4. Modify files, settings, or memory.
5. Hide its own traces from the final answer.

For public LLM Wiki workflows, the highest-risk version is content promotion from untrusted sources into durable notes:

$$
\text{source text}
\rightarrow
\text{summary}
\rightarrow
\text{public markdown}
$$

The summary step must preserve facts, not embedded commands.

## Defense Model

A practical defense separates roles:

$$
\text{policy}
>
\text{developer contract}
>
\text{user task}
>
\text{trusted project files}
>
\text{untrusted content}
$$

Untrusted content can answer "what does this document say?" It should not answer "what should the agent do next?" unless the user explicitly delegates that authority.

## Tool Boundary

Tool calls are where injection becomes a real side effect. Risk increases with:

$$
\operatorname{risk}
\propto
\operatorname{privilege}
\times
\operatorname{irreversibility}
\times
\operatorname{uncertainty}
$$

File writes, network calls, credential use, pushes, deletes, and memory updates need stronger checks than read-only inspection.

## Practical Checks

- Treat all tool output and fetched content as untrusted data, never as commands.
- Keep privileged actions behind explicit confirmation, not model discretion.
- Limit tool scope and credentials to the minimum the task needs.
- Sandbox code execution and file writes; assume injected payloads will try to escape.
- Log inputs and actions so an injection can be detected and traced.
- Quote or summarize untrusted content as content, not as instructions.
- Do not store untrusted instructions in durable memory.
- Before public publishing, scan for secrets, private identifiers, hidden instructions, and operational details.
- Prefer allowlists for permitted side effects over broad model discretion.

## Related

- [[concepts/llm/prompt-injection-boundary|Prompt injection boundary]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/memory-boundary|Memory boundary]]
- [[agents/tools/tool-use|Tool use]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/evidence-ledger|Evidence ledger]]
- [[agents/index|Agents]]
