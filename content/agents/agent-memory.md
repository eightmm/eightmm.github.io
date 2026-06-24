---
title: Agent Memory
tags:
  - agents
  - llm
  - memory
---

# Agent Memory

Agent memory is how an agent carries information beyond a single context window — scratch notes, durable facts, prior decisions, and retrieved documents. Without it, every turn starts blind; with too much, context fills with noise.

## Practical Checks

- Separate short-term working state from durable, reusable facts.
- Store one fact per record with a short description so recall stays selective.
- Prefer retrieval on demand over loading everything into context.
- Verify a recalled fact still holds before acting on it.
- Avoid persisting secrets, private paths, or unpublished results.

## Related

- [[agents/planning|Planning]]
- [[agents/tool-use|Tool use]]
- [[agents/index|Agents]]
- [[concepts/evaluation/index|Evaluation]]
