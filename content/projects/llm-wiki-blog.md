---
title: LLM Wiki Blog
tags:
  - projects
  - llm-wiki
  - quartz
---

# LLM Wiki Blog

This project is the public knowledge-base layer behind the blog: Korean posts provide readable entry points, while English wiki notes provide reusable concepts, paper notes, research maps, and infrastructure references.

## Artifact

The artifact is the public Quartz site itself: a small Korean blog surface connected to a larger English wiki graph.

## Problem

Research notes, paper summaries, agent outputs, and server-operation lessons easily become scattered. A public LLM Wiki gives them a stable structure without exposing private operational details.

## Public Boundary

The public site should contain generalized knowledge, reusable writing formats, and verified public references. It should not expose private infrastructure, unpublished experimental results, internal task names, or collaborator-specific context.

## Design

- Use [[posts/index|Posts]] for Korean narrative writing.
- Use [[posts/blog-writing-guide|Blog writing guide]] and [[posts/topic-roadmap|Topic roadmap]] to keep posts connected to wiki notes.
- Use [[agents/workflows/content-promotion-workflow|Content promotion workflow]] and [[posts/wiki-to-post-workflow|Wiki에서 post로 승격하는 방식]] to decide when a wiki cluster becomes a Korean post.
- Use [[concepts/index|Concepts]] for reusable English notes.
- Use [[papers/index|Papers]] for public paper-specific claims.
- Use [[research/index|Research]] for domain maps.
- Use [[infra/index|Infra]] for generalized operations knowledge.
- Use [[agents/index|Agents]] for workflows involving coding agents and research agents.

## Status

Active. The main work is to keep gateway pages readable while moving reusable definitions, formulas, workflows, and review criteria into English wiki notes.

Lifecycle status: `active`.

## Artifact Release

- Site: released as a public Quartz site.
- Source notes: released when public-safe and verified.
- Private raw notes or agent drafts: not released.
- Build output: generated artifact, not edited manually.

## Checks

- Does every page have a clear role: post, concept, paper, project, infra, agent, or research map?
- Are links bidirectional enough that readers can move from a post into the wiki?
- Are uncertain claims kept out of polished pages until verified?
- Are private paths, accounts, endpoints, and unpublished results absent?
- Are project lifecycle and artifact-release boundaries explicit?

## Related

- [[projects/project-lifecycle|Project lifecycle]]
- [[projects/project-artifact-release|Project artifact release]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
- [[posts/wiki-to-post-workflow|Wiki에서 post로 승격하는 방식]]
- [[inbox/inbox-triage|Inbox triage]]
- [[logs/sanitization-checklist|Sanitization checklist]]
- [[projects/project-note-format|Project note format]]
- [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]]
- [[concepts/index|Concepts]]
