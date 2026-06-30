---
title: LLM Wiki Blog
tags:
  - projects
  - llm-wiki
  - quartz
---

# LLM Wiki Blog

이 project는 블로그 뒤의 public knowledge-base layer입니다. Korean post는 읽기 쉬운 entry point를 제공하고, wiki note는 reusable concept, paper note, research map, infrastructure reference를 제공합니다.

## Artifact

Artifact는 public Quartz site 자체입니다. 작은 Korean blog surface가 더 큰 wiki graph와 연결되는 구조입니다.

## Problem

Research note, paper summary, agent output, server-operation lesson은 쉽게 흩어집니다. Public LLM Wiki는 private operational detail을 노출하지 않고 이것들을 안정적인 구조로 묶습니다.

## Public Boundary

Public site에는 generalized knowledge, reusable writing format, verified public reference만 둡니다. Private infrastructure, unpublished experimental result, internal task name, collaborator-specific context는 노출하지 않습니다.

## Design

- [[posts/index|Posts]]는 Korean narrative writing에 사용합니다.
- [[posts/workflows/blog-writing-guide|Blog writing guide]]와 [[posts/workflows/topic-roadmap|Topic roadmap]]으로 post가 wiki note와 연결되게 합니다.
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]와 [[posts/essays/wiki-to-post-workflow|Wiki에서 post로 승격하는 방식]]으로 wiki cluster가 Korean post가 될 시점을 결정합니다.
- [[concepts/index|Concepts]]는 reusable wiki note에 사용합니다.
- [[papers/index|Papers]]는 public paper-specific claim에 사용합니다.
- [[research/index|Research]]는 domain map에 사용합니다.
- [[infra/index|Infra]]는 generalized operations knowledge에 사용합니다.
- [[agents/index|Agents]]는 coding agent와 research agent를 포함한 workflow에 사용합니다.

## Status

Active. 주요 작업은 public entry page를 읽기 쉽게 유지하면서 reusable definition, formula, workflow, review criteria를 wiki note로 옮기는 것입니다.

Lifecycle status: `active`.

## Artifact Release

- Site: public Quartz site로 released.
- Source notes: public-safe하고 verified된 경우 released.
- Private raw notes 또는 agent draft: not released.
- Build output: generated artifact이며 manually edit하지 않음.

## Checks

- 모든 page에 post, concept, paper, project, infra, agent, research map 중 clear role이 있는가?
- reader가 post에서 wiki로 이동할 수 있을 만큼 link가 bidirectional한가?
- uncertain claim은 verified되기 전까지 polished page에서 제외되는가?
- private path, account, endpoint, unpublished result가 없는가?
- project lifecycle과 artifact-release boundary가 explicit한가?

## Related

- [[projects/project-lifecycle|Project lifecycle]]
- [[projects/project-artifact-release|Project artifact release]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
- [[posts/essays/wiki-to-post-workflow|Wiki에서 post로 승격하는 방식]]
- [[inbox/inbox-triage|Inbox triage]]
- [[logs/sanitization-checklist|Sanitization checklist]]
- [[projects/project-note-format|Project note format]]
- [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]]
- [[concepts/index|Concepts]]
