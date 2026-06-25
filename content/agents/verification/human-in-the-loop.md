---
title: Human in the Loop
tags:
  - agents
  - workflows
  - review
---

# Human in the Loop

Human-in-the-loop design keeps important decisions with a human reviewer while delegating search, drafting, editing, and routine verification to agents.

The key split is:

$$
\operatorname{Decision}
=
\operatorname{HumanReview}
\left(
\operatorname{AgentProposal},
\operatorname{Evidence}
\right)
$$

The agent should make the proposal and evidence cheap to inspect; the human remains responsible for judgment on ambiguous or high-risk choices.

## Good Handoffs

- The agent states what changed and how it was verified.
- The agent separates facts, assumptions, and unresolved questions.
- The human reviews high-risk surfaces such as data, security, dependencies, and public claims.
- The final artifact records enough provenance to revisit the decision.

## Checks

- Which decisions require explicit human approval?
- Is the artifact reviewable without reading the full agent transcript?
- Are uncertain claims marked instead of smoothed over?
- Are public notes sanitized before publication?

## Related

- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[papers/paper-note-format|Paper note format]]
