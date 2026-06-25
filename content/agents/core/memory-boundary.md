---
title: Memory Boundary
tags:
  - agents
  - memory
  - privacy
---

# Memory Boundary

A memory boundary defines what an agent may store, recall, and reuse across tasks. It prevents useful long-term context from becoming a source of stale assumptions, privacy leaks, or task drift.

Memory can be divided into:

$$
M
=
M_{\mathrm{working}}
\cup
M_{\mathrm{durable}}
\cup
M_{\mathrm{external}}
$$

where working memory is task-local, durable memory persists across sessions, and external memory is retrieved from files, docs, databases, or search.

## Boundary Questions

- What is safe to persist?
- What must remain task-local?
- What must never be stored?
- What facts need revalidation before reuse?
- Who is allowed to update durable memory?

## Public Wiki Rule

For this blog, public durable notes should store general concepts, public workflows, and sanitized guidance. They should not store private infrastructure details, credentials, internal task names, unpublished results, or collaborator-specific information.

## Stale Memory Failure

An agent can fail by treating old context as current fact:

$$
\operatorname{risk}
\propto
\operatorname{age}(m)
\times
\operatorname{impact}(m)
\times
(1-\operatorname{verification}(m))
$$

This is not a real calibrated metric, but it captures the operational habit: older and higher-impact memories need stronger verification.

## Checks

- Is the remembered fact public, current, and relevant?
- Is the source authoritative enough for the action?
- Could storing this fact leak private information?
- Does the task need durable memory, or only a local note?
- Is there a deletion or correction path for wrong memories?

## Related

- [[agents/core/agent-memory|Agent memory]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[logs/sanitization-checklist|Sanitization checklist]]
- [[concepts/llm/context-window|Context window]]
