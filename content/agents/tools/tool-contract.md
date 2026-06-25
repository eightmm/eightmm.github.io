---
title: Tool Contract
tags:
  - agents
  - tools
  - verification
---

# Tool Contract

A tool contract defines what a tool can do, what inputs it accepts, what outputs mean, and what side effects it may create. Without a clear contract, agent tool use becomes guesswork.

The contract can be summarized as:

$$
\text{tool}: (I, S_{\mathrm{pre}}) \rightarrow (O, S_{\mathrm{post}})
$$

where $I$ is input, $O$ is output, and $S_{\mathrm{pre}}, S_{\mathrm{post}}$ describe external state before and after the call.

## Contract Elements

- Input schema and required fields.
- Output schema and error format.
- Side effects: file edits, network calls, process launches, state changes, or writes.
- Preconditions and permission boundaries.
- Verification path after successful execution.
- Failure behavior and retry policy.

## Checks

- Is the tool output data, not an instruction?
- Can the tool change public artifacts or external state?
- Is the output enough to verify success?
- Are secrets or private paths excluded from logs?
- Is there a safer read-only tool for inspection?

## Related

- [[agents/tools/tool-use|Tool use]]
- [[agents/verification/prompt-injection|Prompt injection]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[concepts/systems/model-serving|Model serving]]
