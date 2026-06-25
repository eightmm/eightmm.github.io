---
title: Agent Architecture
tags:
  - agents
  - architecture
  - workflows
---

# Agent Architecture

Agent architecture describes the components that turn a model call into a workflow. A useful agent is not only an LLM; it is a loop with state, tools, memory, policy, and verification.

A simple agent can be written as:

$$
a_t = \pi_\theta(s_t, c_t, m_t)
$$

where $s_t$ is task state, $c_t$ is current context, $m_t$ is retrieved memory, and $a_t$ is the next action.

## Components

- Model: generates plans, tool calls, edits, or answers.
- State: tracks goal, constraints, evidence, and progress.
- Context: selected information visible in the current call.
- Tools: actions that inspect or change external state.
- Memory: durable or retrieved information outside the context window.
- Verifier: checks whether the artifact is correct.
- Human boundary: defines where approval or judgment is required.

## Checks

- What state is explicit rather than hidden in conversation history?
- Which tools can create side effects?
- What verifies each side effect?
- What stops the loop?
- Which decisions require human review?

## Related

- [[agents/agent-loop|Agent loop]]
- [[agents/agent-state|Agent state]]
- [[agents/tool-contract|Tool contract]]
- [[agents/context-engineering|Context engineering]]
- [[agents/verification-loop|Verification loop]]
