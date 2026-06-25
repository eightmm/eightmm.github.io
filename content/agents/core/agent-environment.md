---
title: Agent Environment
tags:
  - agents
  - environment
---

# Agent Environment

An agent environment is the external state the agent can observe or change. It can include files, repositories, terminals, web pages, APIs, queues, issue trackers, or generated artifacts.

A simple transition view is:

$$
s_{t+1} = T(s_t, a_t, e_t)
$$

$s_t$ is the agent's working state, $a_t$ is the selected action, $e_t$ is the environment state, and $T$ is the transition caused by the action and observation.

## Environment Types

- Read-only environment: search results, logs, documentation, repository files.
- Editable environment: source files, Markdown notes, configuration, generated artifacts.
- External environment: APIs, CI systems, deployment targets, issue trackers.
- Human environment: approvals, feedback, task constraints, and review decisions.

## Why It Matters

Agents fail when they confuse text context with actual external state. A plan written in conversation does not prove that a file changed, a build passed, or a deployment succeeded. The environment has to be inspected directly.

## Checks

- What external state can the agent observe?
- What external state can the agent change?
- Which actions are read-only and which are side-effecting?
- What evidence proves that the environment changed as intended?
- What state should be ignored because it is generated or out of scope?

## Related

- [[agents/core/agent-loop|Agent loop]]
- [[agents/core/agent-state|Agent state]]
- [[agents/core/action-space|Action space]]
- [[agents/tools/tool-use|Tool use]]
- [[agents/verification/verification-loop|Verification loop]]
