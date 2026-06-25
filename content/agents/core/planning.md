---
title: Planning
tags:
  - agents
  - llm
  - planning
---

# Planning

Planning is the step where an agent decomposes a goal into ordered, verifiable subtasks before acting. Good plans bound blast radius, surface assumptions, and make progress checkable.

## Practical Checks

- State the goal, constraints, and success criteria before the first action.
- Break work into steps that each have a concrete verification.
- Re-plan when a step fails or new information contradicts the assumptions.
- Keep the plan visible so drift from the original goal is detectable.
- Stop and ask when intent is ambiguous rather than guessing.

## Related

- [[agents/core/agent-loop|Agent loop]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/tools/tool-use|Tool use]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[agents/index|Agents]]
