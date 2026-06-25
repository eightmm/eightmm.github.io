---
title: Action Space
tags:
  - agents
  - planning
---

# Action Space

The action space is the set of actions an agent can choose from at a given step. It includes both natural-language actions, such as asking a question, and tool actions, such as reading a file, editing a note, running a command, or opening a pull request.

Formally:

$$
a_t \in \mathcal{A}(s_t, b_t)
$$

$a_t$ is the action at step $t$, $\mathcal{A}$ is the available action set, $s_t$ is task state, and $b_t$ is the permission or boundary condition.

## Action Classes

- Observe: inspect files, logs, pages, or command output.
- Transform: edit source, rewrite notes, summarize evidence, or refactor text.
- Execute: run builds, tests, formatters, deployments, or scripts.
- Ask: request missing constraints, approval, or domain judgment.
- Stop: report completion, failure, or blocked state with evidence.

## Constraints

Not every possible action should be available at every step. Side-effecting actions need a clear purpose and a verification path. Destructive, expensive, private, or externally visible actions need stronger boundaries.

## Checks

- Is the next action in the allowed action space?
- Is a read-only action enough before editing?
- Does the action have a bounded result?
- What verifier will inspect the action result?
- Should the agent ask before taking the action?

## Related

- [[agents/core/planning|Planning]]
- [[agents/core/agent-environment|Agent environment]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[agents/verification/verification-loop|Verification loop]]
