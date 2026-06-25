---
title: Content Promotion Workflow
tags:
  - agents
  - workflows
  - wiki
  - publishing
---

# Content Promotion Workflow

Content promotion is the workflow that turns raw notes into durable wiki pages, Korean posts, paper notes, project notes, or public logs. It keeps the site from becoming either an unreadable note dump or a shallow blog without reusable knowledge.

A useful promotion path is:

$$
\text{raw input}
\rightarrow
\text{inbox}
\rightarrow
\text{canonical wiki note}
\rightarrow
\text{synthesis}
\rightarrow
\text{post or project note}
$$

## Destinations

- Inbox: unverified candidates, daily briefs, and follow-up queues.
- Concept: reusable definitions, equations, checklists, and evaluation rules.
- Paper: paper-specific claims, evidence, artifacts, limitations, and reproduction plans.
- Project: public artifact, design, verification, status, and next work.
- Infra: public operational lesson or systems pattern.
- Log: cleaned work record, incident note, or experiment summary.
- Log promotion: use [[logs/public-log-taxonomy|Public log taxonomy]] and [[logs/log-promotion-rule|Log promotion rule]] before leaving a cleaned record as a standalone log.
- Post: Korean narrative that gives context, reading order, and interpretation.

## Promotion Criteria

Use a page's strongest role as the destination:

$$
D(n)
=
\arg\max_{d \in \mathcal{D}}
\mathrm{fit}(n,d)
$$

where $n$ is a candidate note, $\mathcal{D}$ is the destination set, and $\mathrm{fit}$ scores whether the note is reusable, paper-specific, artifact-specific, operational, narrative, or still unverified.

## When To Write A Post

A Korean post is ready when:

- Several wiki notes already exist.
- The reader needs a map or point of view, not another isolated definition.
- There is a clear question that the post answers.
- The post can link to concepts, papers, projects, or infra notes instead of copying them.
- The content passes [[inbox/publishing-gate|Publishing gate]] and [[logs/sanitization-checklist|Sanitization checklist]].

## Agent Role

An agent should:

- Classify the raw input.
- Prefer updating existing pages over creating weak duplicates.
- Create small stubs only when they are linked and useful.
- Add formulas where they clarify objectives, metrics, update rules, or decision criteria.
- Mark missing facts as `to verify`.
- Run link, privacy, and build checks before committing.

## Checks

- Does the destination match the note's strongest role?
- Does the page have inbound and outbound links?
- Is the canonical explanation in the wiki, not duplicated across posts?
- Does a Korean post provide context and reading order?
- Are private details removed before public promotion?
- Is there an explicit next action for items left in the inbox?
- If the candidate is a log, is its taxonomy and promotion destination explicit?

## Related

- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[agents/workflows/agent-runbook|Agent runbook]]
- [[inbox/inbox-triage|Inbox triage]]
- [[inbox/curation-queue|Curation queue]]
- [[inbox/publishing-gate|Publishing gate]]
- [[posts/blog-writing-guide|Blog writing guide]]
- [[posts/wiki-to-post-workflow|Wiki에서 post로 승격하는 방식]]
- [[projects/llm-wiki-blog|LLM Wiki blog]]
- [[logs/public-log-taxonomy|Public log taxonomy]]
- [[logs/log-promotion-rule|Log promotion rule]]
