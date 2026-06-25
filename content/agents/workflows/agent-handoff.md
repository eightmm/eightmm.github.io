---
title: Agent Handoff
tags:
  - agents
  - workflows
  - handoff
---

# Agent Handoff

Agent handoff transfers a task, artifact, or decision from one agent or role to another. A good handoff makes the next agent productive without requiring it to trust hidden context.

A handoff should include:

$$
H
=
\{\text{goal},\text{scope},\text{state},\text{artifacts},\text{evidence},\text{risks},\text{next step}\}
$$

## Handoff Contents

- Goal: what still needs to be done.
- Scope: files, topics, or systems that are in bounds.
- Current state: what changed and what remains.
- Artifacts: paths, commits, outputs, or drafts.
- Evidence: checks already run and their results.
- Risks: known uncertainty, failed checks, or sensitive boundaries.
- Next step: the smallest useful continuation.

## Common Handoff Types

- Research handoff: paper candidates and verification status.
- Coding handoff: patch summary, tests, and risk areas.
- Review handoff: findings, severity, and evidence.
- Operations handoff: failure record, last good state, and recovery plan.
- Wiki handoff: new pages, broken links, stubs, and curation queue.

## Checks

- Can the receiver verify the claimed state?
- Are private details removed or generalized?
- Are completed and pending steps separated?
- Are skipped checks clearly marked?
- Is the next step concrete enough to execute?

## Related

- [[agents/workflows/agent-orchestration|Agent orchestration]]
- [[agents/core/task-decomposition|Task decomposition]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[concepts/systems/failure-recovery|Failure recovery]]
