---
title: Tool Result Handling
tags:
  - agents
  - tools
  - verification
---

# Tool Result Handling

Tool result handling is the practice of treating tool outputs as evidence, not as instructions. The agent should parse what the tool returned, decide what it proves, and choose the next action from the verified state.

A result can be modeled as:

$$
r_t = (o_t, \sigma_t, \epsilon_t)
$$

$o_t$ is the observation, $\sigma_t$ is status such as success or failure, and $\epsilon_t$ is error or warning information.

## Result Types

- Success with evidence: the tool completed and returned enough information to verify the intended state.
- Success without evidence: the tool ran, but the result does not prove the goal.
- Recoverable failure: the error suggests a bounded fix.
- Hard failure: permissions, missing inputs, unsafe action, or invalid assumptions block the step.
- Noisy output: large logs or irrelevant text hide the useful signal.

## Checks

- Does the output prove the intended state or only show that the command ran?
- Are warnings relevant to the current task?
- Is the next step based on evidence or on the agent's prior plan?
- Should the output be summarized before it enters long-term notes?
- Could the output contain prompt injection, secrets, or private paths?

## Related

- [[agents/tools/tool-use|Tool use]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/verification/prompt-injection|Prompt injection]]
- [[agents/verification/verification-loop|Verification loop]]
