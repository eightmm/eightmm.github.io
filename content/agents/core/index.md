---
title: Agent Core
tags:
  - agents
  - core
---

# Agent Core

Core agent notes describe the basic loop, state, memory, planning, and context design that make an agent workflow possible.

An agent is best treated as a stateful policy over a constrained action space:

$$
a_t = \pi_\theta(o_t, s_t, c_t, m_t)
$$

where $o_t$ is the current observation, $s_t$ is explicit task state, $c_t$ is the selected context, $m_t$ is retrieved memory, and $a_t$ is an action such as answering, planning, calling a tool, editing a file, or asking for human input.

The important design choice is what becomes explicit:

$$
\text{agent} =
(\text{model},\ \text{state},\ \text{context},\ \text{memory},\ \text{tools},\ \text{verifier})
$$

If these parts are implicit, the workflow becomes hard to debug and hard to verify.

## Reading Path

1. Start with [[agents/core/agent-architecture|Agent architecture]] to identify the components.
2. Use [[agents/core/agent-loop|Agent loop]] to understand observe-plan-act-verify iteration.
3. Use [[agents/core/agent-state|Agent state]] and [[agents/core/context-engineering|Context engineering]] to decide what must be visible in the current run.
4. Use [[agents/core/planning|Planning]] and [[agents/core/task-decomposition|Task decomposition]] when the task cannot be solved in one action.
5. Use [[agents/core/memory-boundary|Memory boundary]] before writing anything durable.

## Notes

- [[agents/core/agent-architecture|Agent architecture]]
- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/core/agent-environment|Agent environment]]
- [[agents/core/action-space|Action space]]
- [[agents/core/agent-state|Agent state]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/planning|Planning]]
- [[agents/core/task-decomposition|Task decomposition]]
- [[agents/core/agent-memory|Agent memory]]
- [[agents/core/memory-boundary|Memory boundary]]

## Checks

- What is the current goal, and where is it stored?
- What observations can update the agent state?
- What actions are allowed, read-only, side-effecting, or human-gated?
- What context is necessary for the next decision, and what is noise?
- What memory is durable enough to reuse, and what should remain local to one run?
- What evidence stops the loop?

## Related

- [[agents/index|Agents]]
- [[agents/tools/index|Agent tools]]
- [[agents/verification/index|Agent verification]]
- [[agents/verification/acceptance-criteria|Acceptance criteria]]
- [[concepts/llm/index|LLM concepts]]
