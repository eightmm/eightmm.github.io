---
title: Agent State
tags:
  - agents
  - workflows
  - memory
---

# Agent State

Agent state is the structured information an agent uses to decide what to do next. It should not be treated as whatever happens to remain in the conversation.

A useful state can be modeled as:

$$
s_t = (g, C, E_t, P_t, A_t)
$$

where $g$ is the goal, $C$ is constraints, $E_t$ is accumulated evidence, $P_t$ is the plan, and $A_t$ is the action history.

## State Types

- Goal state: objective, scope, success criteria, and stop conditions.
- Environment state: files, command outputs, browser state, repository status, or external artifacts.
- Evidence state: facts gathered from tools or primary sources.
- Plan state: pending, active, and completed steps.
- Risk state: approvals, destructive actions, secrets, and public/private boundaries.

## Checks

- Is the goal still the newest user request?
- Is the current plan based on evidence or stale memory?
- Are assumptions separated from observed facts?
- Are private details excluded from durable public notes?
- Does state survive context changes without becoming misleading?

## Related

- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/agent-memory|Agent memory]]
- [[agents/core/planning|Planning]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
