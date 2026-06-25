---
title: Tool Calling
tags:
  - llm
  - agents
  - tool-use
---

# Tool Calling

Tool calling lets a language model request an external operation through a structured interface. The model proposes a call; the harness validates and executes it; the result returns as data.

A tool call can be represented as:

$$
\operatorname{call}
=
(\text{name}, \text{arguments}, \text{schema}, \text{permission})
$$

The loop is:

$$
\text{model output}
\rightarrow
\text{validate}
\rightarrow
\text{execute}
\rightarrow
\text{return observation}
\rightarrow
\text{continue}
$$

## Contract

- Name: what operation is being requested.
- Arguments: typed fields with constraints.
- Permission: read-only, write, side-effecting, or gated.
- Result shape: concise output returned to the model.
- Error shape: explicit failure rather than silent fallback.

## Failure Modes

- Invalid arguments.
- Tool chosen for the wrong task.
- Side effects without confirmation.
- Prompt injection through tool output.
- Oversized results that consume context.
- Model treats tool output as trusted instruction instead of data.

## Checks

- Is the tool needed, or can the model answer from given evidence?
- Are arguments validated before execution?
- Are side effects gated or reversible?
- Is output concise and source-attributed?
- Is untrusted output isolated from privileged instructions?

## Related

- [[agents/tools/tool-use|Tool use]]
- [[agents/tools/tool-contract|Tool contract]]
- [[concepts/llm/structured-output|Structured output]]
- [[concepts/llm/prompt-injection-boundary|Prompt injection boundary]]
- [[agents/verification/verification-loop|Verification loop]]
