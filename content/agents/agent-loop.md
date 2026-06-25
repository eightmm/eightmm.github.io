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

## Related

- [[agents/planning|Planning]]
- [[agents/tool-use|Tool use]]
- [[agents/verification-loop|Verification loop]]
- [[agents/context-engineering|Context engineering]]
