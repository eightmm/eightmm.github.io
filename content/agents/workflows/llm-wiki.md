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

LLM Wiki는 retrieval, synthesis, human review를 돕는 public linked knowledge base입니다. 핵심 단위는 hidden vector index가 아니라 explicit link를 가진 짧은 Markdown note입니다.

## Goals

- reusable concept는 [[concepts/index|Concepts]]에 둡니다.
- paper-specific claim은 [[papers/index|Papers]]에 둡니다.
- uncurated daily candidate는 review 전까지 [[inbox/index|Inbox]]에 둡니다.
- active research context는 [[research/index|Research]]에 둡니다.
- workflow, tool, human-AI collaboration을 설명하는 agent note는 [[agents/index|Agents]]에 둡니다.
- operational note는 private system을 노출하지 않고 [[infra/index|Infra]]에 둡니다.
- public project summary는 [[projects/index|Projects]]에, cleaned work record는 [[logs/index|Public logs]]에 둡니다.

## Note 형태

- topic 하나에 page 하나를 둡니다.
- fact, assumption, open question의 boundary를 명확히 합니다.
- adjacent concept, paper, project로 wikilink를 둡니다.
- evidence-grounded claim을 씁니다. Support되는 statement는 source, concept, log, paper note로 연결하고, uncertain statement는 `to verify`로 표시합니다.
- private endpoint, account name, credential, unpublished metric, internal project identifier를 넣지 않습니다.
- 한국어 post는 읽기 쉬운 entry point를 제공하고, wiki note는 reusable definition과 check를 담습니다.
- reusable note는 post나 paper note의 anchor가 되기 전에 [[concepts/wiki-note-quality-gate|Wiki note quality gate]]를 통과해야 합니다.
- raw material이 inbox item, concept note, paper note, project note, infra note, public log, 한국어 post 중 무엇이 될지 [[agents/workflows/content-promotion-workflow|Content promotion workflow]]로 결정합니다.

## Related

- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
- [[concepts/wiki-note-quality-gate|Wiki note quality gate]]
- [[projects/llm-wiki-blog|LLM Wiki blog]]
- [[posts/blog-writing-guide|Blog writing guide]]
- [[posts/wiki-to-post-workflow|Wiki에서 post로 승격하는 방식]]
- [[posts/topic-roadmap|Topic roadmap]]
- [[concepts/llm/index|LLM concepts]]
- [[concepts/llm/evidence-grounded-generation|Evidence-grounded generation]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[concepts/llm/chunking|Chunking]]
- [[concepts/llm/hybrid-retrieval|Hybrid retrieval]]
- [[concepts/llm/query-rewriting|Query rewriting]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/agent-loop|Agent loop]]
- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[infra/hpc/slurm|Slurm]]
- [[ai/generative-models|Generative models]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[agents/workflows/coding-agents|Coding agents]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
