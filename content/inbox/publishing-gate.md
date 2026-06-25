---
title: Publishing Gate
tags:
  - inbox
  - publishing
  - workflows
---

# Publishing Gate

A publishing gate is the final check before a raw or private working note becomes public content. It verifies that the note is useful, sourced, linked, and safe to publish.

The gate asks whether a note satisfies:

$$
\text{publishable}
=
\text{public-safe}
\land
\text{verified}
\land
\text{linked}
\land
\text{useful}
$$

## Gate Checklist

- Public-safe: no secrets, private infrastructure, account details, collaborator details, internal task names, or unpublished sensitive results.
- Verified: metadata, claims, commands, or source facts are checked or marked `to verify`.
- Linked: the note connects to relevant concepts, papers, projects, infra, logs, or posts.
- Useful: the note teaches a reusable concept, records a public decision, or supports a reader path.
- Scoped: the note fits the site's AI, Bio-AI, Infra, Agents, Papers, Projects, or Research map.

## Before Publishing

- Run a sensitive-information scan.
- Check wikilinks.
- Build the site.
- Keep generated `public/` output out of manual edits.
- Commit and push only task-relevant files.

## Checks

- Would the note make sense to an outside reader?
- Is private context generalized enough?
- Are uncertain statements labeled clearly?
- Is this better as an update to an existing page?
- Is there a follow-up owner or next page if the note remains incomplete?

## Related

- [[logs/sanitization-checklist|Sanitization checklist]]
- [[inbox/curation-queue|Curation queue]]
- [[inbox/inbox-triage|Inbox triage]]
- [[posts/blog-writing-guide|블로그 글 작성 가이드]]
