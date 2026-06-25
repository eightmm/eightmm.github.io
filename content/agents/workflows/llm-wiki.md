---
title: LLM Wiki
aliases:
  - research/llm-wiki
tags:
  - llm
  - knowledge-base
  - wiki
---

# LLM Wiki

An LLM Wiki is a public, linked knowledge base that supports retrieval, synthesis, and human review. The core unit is a short Markdown note with explicit links, not a hidden vector index.

## Goals

- Keep reusable concepts in [[concepts/index|Concepts]].
- Keep paper-specific claims in [[papers/index|Papers]].
- Keep uncurated daily candidates in [[inbox/index|Inbox]] until reviewed.
- Keep active research context in [[research/index|Research]].
- Keep agent notes in [[agents/index|Agents]] when they describe workflows, tools, and human-AI collaboration.
- Keep operational notes in [[infra/index|Infra]] without exposing private systems.
- Keep public project summaries in [[projects/index|Projects]] and cleaned work records in [[logs/index|Public logs]].

## Note shape

- One page per topic.
- Clear boundaries between facts, assumptions, and open questions.
- Wikilinks for adjacent concepts, papers, and projects.
- Evidence-grounded claims: supported statements link to sources, concepts, logs, or paper notes; uncertain statements are marked `to verify`.
- No private endpoints, account names, credentials, unpublished metrics, or internal project identifiers.
- Korean posts should provide readable entry points; English wiki notes should hold reusable definitions and checks.
- Use [[agents/workflows/content-promotion-workflow|Content promotion workflow]] to decide whether raw material becomes an inbox item, concept note, paper note, project note, infra note, public log, or Korean post.

## Related

- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
- [[projects/llm-wiki-blog|LLM Wiki blog]]
- [[posts/blog-writing-guide|Blog writing guide]]
- [[posts/wiki-to-post-workflow|Wiki에서 post로 승격하는 방식]]
- [[posts/topic-roadmap|Topic roadmap]]
- [[concepts/llm/index|LLM concepts]]
- [[concepts/llm/evidence-grounded-generation|Evidence-grounded generation]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/agent-loop|Agent loop]]
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[infra/hpc/slurm|Slurm]]
- [[ai/generative-models|Generative models]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[agents/workflows/coding-agents|Coding agents]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
