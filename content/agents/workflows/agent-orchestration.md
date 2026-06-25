---
title: Agent Orchestration
tags:
  - agents
  - workflows
  - multi-agent
---

# Agent Orchestration

Agent orchestration coordinates multiple model calls, tools, roles, or agents into one workflow. It is useful when a task naturally separates into discovery, drafting, implementation, review, and verification.

A workflow can be represented as a directed graph:

$$
G = (V, E)
$$

where nodes $V$ are steps or roles, and edges $E$ pass artifacts, evidence, or decisions.

## Patterns

- Pipeline: one step produces input for the next step.
- Supervisor-worker: one controller delegates bounded tasks.
- Reviewer loop: a second pass checks the first artifact.
- Debate or council: independent agents produce opinions before synthesis.
- Human gate: a human approves high-risk transitions.

## Checks

- What artifact passes between steps?
- Are agents independent or sharing the same context bias?
- Who resolves conflicts?
- Is each step verifiable without trusting another agent's confidence?
- Does orchestration add value beyond a single well-scoped agent loop?

## Related

- [[agents/workflows/agent-handoff|Agent handoff]]
- [[agents/workflows/agent-runbook|Agent runbook]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[agents/verification/verification-loop|Verification loop]]
- [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]]
