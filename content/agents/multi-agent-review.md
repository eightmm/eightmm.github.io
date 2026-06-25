---
title: Multi-Agent Review
tags:
  - agents
  - llm
  - review
---

# Multi-Agent Review

Multi-agent review uses more than one agent — often different models or roles — to independently inspect a change before it is accepted. Independent passes catch errors a single agent rationalizes away.

## Practical Checks

- Give each reviewer the diff plus the goal, not a summary that hides detail.
- Prefer adversarial prompts ("try to refute this") over confirmation.
- Treat agreement across independent models as weak evidence, not proof.
- Gate high-risk surfaces — APIs, auth, data pipelines, training, deps — behind review.
- Record verdicts so a decision can be re-examined later.

## Related

- [[agents/human-in-the-loop|Human in the loop]]
- [[agents/agent-evaluation|Agent evaluation]]
- [[agents/verification-loop|Verification loop]]
- [[agents/planning|Planning]]
- [[agents/coding-agents|Coding agents]]
- [[agents/index|Agents]]
