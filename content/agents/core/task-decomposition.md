---
title: Task Decomposition
tags:
  - agents
  - planning
  - workflows
---

# Task Decomposition

Task decomposition breaks a broad goal into bounded units that an agent can execute and verify. It is the bridge between an ambiguous user request and concrete tool calls.

A task can be represented as a set of subtasks:

$$
T
=
\{t_1,t_2,\ldots,t_n\}
$$

with dependencies:

$$
E
\subseteq
T \times T
$$

where $(t_i,t_j)\in E$ means $t_j$ depends on $t_i$.

## Good Subtasks

- Have a clear artifact.
- Have an explicit success check.
- Fit within a small context and tool scope.
- Avoid mixing discovery, editing, and verification in one opaque step.
- Make failure local enough to diagnose.

## Decomposition Patterns

- Discover -> edit -> verify -> summarize.
- Draft -> review -> revise -> publish.
- Ingest -> sanitize -> link -> curate.
- Plan -> delegate -> admit patch -> verify.
- Batch large repetitive work by folder, topic, or risk level.

## Checks

- Is each subtask independently verifiable?
- Does any subtask need user approval before action?
- Are dependencies explicit?
- Is there one owner for conflict resolution?
- Can the plan be shortened without losing correctness?

## Related

- [[agents/core/planning|Planning]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/workflows/agent-handoff|Agent handoff]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
