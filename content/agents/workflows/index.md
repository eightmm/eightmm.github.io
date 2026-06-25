---
title: Agent Workflows
tags:
  - agents
  - workflows
---

# Agent Workflows

Workflow notes describe practical agent use cases: coding, paper review, orchestration, multi-agent review, and wiki maintenance.

An agent workflow is a repeatable path from input material to a checked artifact:

$$
\text{input}
\rightarrow
\text{triage}
\rightarrow
\text{plan}
\rightarrow
\text{act}
\rightarrow
\text{verify}
\rightarrow
\text{publish or hand off}
$$

The same agent architecture can support different workflows, but each workflow needs its own acceptance criteria, side-effect boundary, and verification ladder.

## Workflow Families

- Coding: inspect source, edit narrowly, run checks, commit, and push.
- Paper briefs: ingest candidates, mark unverified metadata, promote selected notes.
- LLM Wiki maintenance: split raw material into concepts, papers, projects, logs, and posts.
- Multi-agent review: collect independent reads before accepting a risky change.
- Handoff: preserve state, evidence, and open decisions for another run or human reviewer.

## Notes

- [[agents/workflows/coding-agents|Coding agents]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[agents/workflows/agent-orchestration|Agent orchestration]]
- [[agents/workflows/agent-handoff|Agent handoff]]
- [[agents/workflows/agent-runbook|Agent runbook]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]

## Checks

- What artifact should exist at the end?
- What side effects are allowed: file edit, commit, push, issue, job, or publication?
- What evidence proves the artifact is correct enough?
- What material remains inbox, what becomes a wiki note, and what becomes a Korean post?
- What should be handed to a human before public release?

## Related

- [[agents/index|Agents]]
- [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]]
- [[projects/llm-wiki-blog|LLM Wiki blog]]
- [[agents/verification/index|Agent verification]]
- [[inbox/index|Inbox]]
- [[logs/index|Public logs]]
