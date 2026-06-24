---
title: Coding Agents
tags:
  - agents
  - coding
  - research-engineering
---

# Coding Agents

Coding agents are LLM-based tools that can inspect a codebase, edit files, run commands, and report results. They are most useful when the task has a clear goal, local verification path, and bounded blast radius.

## Good Uses

- Refactor a small module with tests.
- Draft documentation from existing code.
- Add narrow features to a known codebase.
- Run repetitive checks across files.
- Review implementation risks before a commit.

## Weak Spots

- Vague product direction.
- Hidden data or environment assumptions.
- Security-sensitive changes without explicit review.
- Large dependency, API, schema, or training changes without a written plan.

## Verification Habit

Every agent-assisted change should end with a concrete check: build, unit test, lint, smoke test, or manual review. The useful question is not whether the agent sounded confident, but whether the artifact is correct.

## Related

- [[agents/index|Agents]]
- [[research/llm-wiki|LLM Wiki]]
- [[projects/index|Projects]]
- [[infra/index|Infra]]
