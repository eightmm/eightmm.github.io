---
title: LLM Wiki Blog
tags:
  - projects
  - llm-wiki
  - quartz
---

# LLM Wiki Blog

This project is the public knowledge-base layer behind the blog: Korean posts provide readable entry points, while English wiki notes provide reusable concepts, paper notes, research maps, and infrastructure references.

## Problem

Research notes, paper summaries, agent outputs, and server-operation lessons easily become scattered. A public LLM Wiki gives them a stable structure without exposing private operational details.

## Design

- Use [[posts/index|Posts]] for Korean narrative writing.
- Use [[posts/blog-writing-guide|Blog writing guide]] and [[posts/topic-roadmap|Topic roadmap]] to keep posts connected to wiki notes.
- Use [[concepts/index|Concepts]] for reusable English notes.
- Use [[papers/index|Papers]] for public paper-specific claims.
- Use [[research/index|Research]] for domain maps.
- Use [[infra/index|Infra]] for generalized operations knowledge.
- Use [[agents/index|Agents]] for workflows involving coding agents and research agents.

## Checks

- Does every page have a clear role: post, concept, paper, project, infra, agent, or research map?
- Are links bidirectional enough that readers can move from a post into the wiki?
- Are uncertain claims kept out of polished pages until verified?
- Are private paths, accounts, endpoints, and unpublished results absent?

## Related

- [[agents/llm-wiki|LLM Wiki]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[agents/context-engineering|Context engineering]]
- [[agents/paper-brief-workflow|Paper brief workflow]]
- [[inbox/inbox-triage|Inbox triage]]
- [[logs/sanitization-checklist|Sanitization checklist]]
- [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]]
- [[concepts/index|Concepts]]
