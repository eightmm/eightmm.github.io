---
title: Agent Loop
tags:
  - agents
  - llm
  - workflows
---

# Agent Loop

An agent loop is the repeated cycle of observing state, deciding the next action, using tools, and verifying the result. It turns a language model from a single-response system into a workflow participant.

A generic loop is:

$$
s_{t+1}
= \operatorname{Update}
\left(
s_t,\,
a_t,\,
o_t
\right)
$$

where $s_t$ is the working state, $a_t$ is the chosen action, and $o_t$ is the observation returned by a tool or environment.

A policy chooses the next action from state:

$$
a_t \sim \pi_\theta(a\mid s_t, g, C)
$$

where $g$ is the goal and $C$ is the [[agents/core/agent-operating-contract|agent operating contract]]. The contract constrains which actions are allowed and which checks prove completion.

## Steps

1. Read the goal, constraints, and current state.
2. Plan a bounded next action.
3. Use a tool or edit an artifact.
4. Observe the result.
5. Verify against the goal.
6. Re-plan or stop.

## Checks

- Is the next action grounded in current evidence?
- Does each side effect have a verification path?
- Does the loop stop when the goal is achieved or blocked?
- Are tool outputs treated as data rather than trusted instructions?
- Is the loop making progress toward the original goal rather than redefining success?
- Does the state include current artifacts, not only memory of prior steps?

## Failure Modes

- Acting from stale context without re-reading the current state.
- Treating tool output, search snippets, or generated text as trusted instructions.
- Repeating the same failed action without changing evidence or plan.
- Reporting completion from plausibility rather than verification.

## Related

- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[agents/core/planning|Planning]]
- [[agents/core/agent-environment|Agent environment]]
- [[agents/core/action-space|Action space]]
- [[agents/tools/tool-use|Tool use]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/core/context-engineering|Context engineering]]
