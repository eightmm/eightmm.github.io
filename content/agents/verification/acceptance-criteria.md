---
title: Acceptance Criteria
tags:
  - agents
  - verification
---

# Acceptance Criteria

Acceptance criteria define what must be true before an agent can claim a task is complete. They turn a vague goal into checkable conditions.

For a task goal $G$, define criteria:

$$
\mathcal{C}(G) = \{c_1, c_2, \ldots, c_k\}
$$

Completion requires evidence for every criterion:

$$
\operatorname{done}(G)
= \bigwedge_{i=1}^{k} \operatorname{verified}(c_i)
$$

## Good Criteria

- Observable: a file, build output, test result, rendered page, or review decision can prove it.
- Specific: the condition is narrow enough to check.
- Complete: the set covers correctness, safety, and publication constraints.
- Current: evidence comes from the current environment, not memory of earlier work.
- Public-safe: criteria include privacy and sanitization when artifacts are published.

## Examples

- A Markdown page exists at the expected path.
- All internal wikilinks resolve.
- `npx quartz build` succeeds.
- Generated public pages do not expose private infrastructure details.
- A commit was pushed to the expected branch.

## Checks

- What evidence proves each criterion?
- Are tests broad enough for the claim being made?
- Is skipped verification reported explicitly?
- Are generated files excluded from manual edits?
- Does the final answer distinguish changed, verified, and not verified items?

## Related

- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[agents/workflows/agent-runbook|Agent runbook]]
- [[logs/sanitization-checklist|Sanitization checklist]]
