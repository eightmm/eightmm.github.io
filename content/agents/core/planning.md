---
title: Planning
tags:
  - agents
  - llm
  - planning
---

# Planning

Planning is the step where an agent decomposes a goal into ordered, verifiable subtasks before acting. Good plans bound blast radius, surface assumptions, and make progress checkable.

A useful plan maps each step to evidence:

$$
P = \{(s_i, v_i)\}_{i=1}^{k}
$$

where $s_i$ is a step and $v_i$ is the check that can verify it. A step without a verification path is usually a guess, not a plan.

## Plan Shape

- State the current evidence.
- Name assumptions that could be wrong.
- Choose a small next action.
- Attach a verification check to that action.
- Re-plan when new evidence changes the problem.

## Practical Checks

- State the goal, constraints, and success criteria before the first action.
- Break work into steps that each have a concrete verification.
- Re-plan when a step fails or new information contradicts the assumptions.
- Keep the plan visible so drift from the original goal is detectable.
- Stop and ask when intent is ambiguous rather than guessing.
- Do not let the plan become a substitute for inspecting files, running checks, or changing the artifact.

## Related

- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/core/task-decomposition|Task decomposition]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/tools/tool-use|Tool use]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[agents/index|Agents]]
