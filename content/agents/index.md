---
title: Agents
tags:
  - agents
  - llm
  - workflows
---

# Agents

Agent notes collect public explanations of AI agents, coding assistants, research workflows, and tool-use patterns.

This section should stay practical: what the agent is useful for, where it fails, how to verify its work, and how it fits into research or infrastructure work.

The local taxonomy is:

$$
\text{agent note}
\rightarrow
\{\text{core},\ \text{tools},\ \text{workflows},\ \text{verification}\}
$$

Use the category that explains the durable idea, not the product name that happened to expose it.

## Sections

| Section | Use For |
| --- | --- |
| [[agents/core/index|Agent Core]] | architecture, loop, state, memory, planning |
| [[agents/tools/index|Agent Tools]] | tool contracts, side effects, result handling |
| [[agents/workflows/index|Agent Workflows]] | coding, paper briefs, orchestration, handoff |
| [[agents/verification/index|Agent Verification]] | acceptance criteria, evidence, audits, evaluation |

## Routing Guide

| If the note is about... | Put it in |
| --- | --- |
| model state, memory, planning, context | [[agents/core/index|Agent Core]] |
| schemas, side effects, permissions, outputs | [[agents/tools/index|Agent Tools]] |
| repeatable use cases and handoffs | [[agents/workflows/index|Agent Workflows]] |
| acceptance criteria, evidence, audits, safety checks | [[agents/verification/index|Agent Verification]] |
| broad reader-facing stories | [[posts/index|Posts]] |

## Core Concepts

| Group | Notes |
| --- | --- |
| Core model | [[agents/core/agent-architecture|Agent architecture]], [[agents/core/agent-operating-contract|Agent operating contract]], [[agents/core/agent-loop|Agent loop]] |
| Environment and state | [[agents/core/agent-environment|Agent environment]], [[agents/core/action-space|Action space]], [[agents/core/agent-state|Agent state]] |
| Context and memory | [[agents/core/context-engineering|Context engineering]], [[agents/core/agent-memory|Agent memory]], [[agents/core/memory-boundary|Memory boundary]] |
| Planning | [[agents/core/planning|Planning]], [[agents/core/task-decomposition|Task decomposition]] |
| LLM interface | [[concepts/llm/prompting|Prompting]], [[concepts/llm/structured-output|Structured output]] |
| Tools | [[agents/tools/tool-use|Tool use]], [[agents/tools/tool-contract|Tool contract]], [[agents/tools/tool-result-handling|Tool result handling]] |
| Verification | [[agents/verification/verification-loop|Verification loop]], [[agents/verification/acceptance-criteria|Acceptance criteria]], [[agents/verification/evidence-ledger|Evidence ledger]] |
| Review and safety | [[agents/verification/completion-audit|Completion audit]], [[agents/verification/reflection-and-critique|Reflection and critique]], [[agents/verification/agent-evaluation|Agent evaluation]], [[agents/verification/human-in-the-loop|Human in the loop]], [[agents/verification/prompt-injection|Prompt injection]] |

## Workflows

| Workflow | Use For |
| --- | --- |
| [[agents/workflows/coding-agents|Coding agents]] | implementation and repository work |
| [[agents/workflows/paper-brief-workflow|Paper brief workflow]] | paper intake and synthesis |
| [[agents/workflows/agent-orchestration|Agent orchestration]] | multiple agents or tools |
| [[agents/workflows/agent-handoff|Agent handoff]] | passing state between workers |
| [[agents/workflows/agent-runbook|Agent runbook]] | repeatable operating procedure |
| [[agents/workflows/multi-agent-review|Multi-agent review]] | independent verification |
| [[agents/workflows/llm-wiki|LLM Wiki]] | knowledge-base maintenance |
| [[agents/workflows/content-promotion-workflow|Content promotion workflow]] | inbox to note/post/project promotion |
| [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]] | project-level paper workflow |

## Learning and Feedback

| Method | Link |
| --- | --- |
| Imitation | [[concepts/learning/imitation-learning|Imitation learning]] |
| Reinforcement | [[concepts/learning/reinforcement-learning|Reinforcement learning]] |
| Reward modeling | [[concepts/learning/reward-modeling|Reward modeling]] |
| Preference objective | [[concepts/learning/preference-optimization|Preference optimization]] |

## Writing Rules

- Explain workflows with generic examples.
- Keep vendor-specific claims dated or scoped.
- Do not publish private prompts, private repo details, credentials, internal task names, or unreleased work.
- Prefer concrete verification habits over broad productivity claims.

## Related

| Area | Link |
| --- | --- |
| Knowledge workflow | [[agents/workflows/llm-wiki|LLM Wiki]], [[inbox/index|Inbox]] |
| LLM behavior | [[concepts/llm/hallucination-grounding|Hallucination and grounding]], [[concepts/architectures/transformer|Transformer]] |
| Public work records | [[projects/index|Projects]], [[logs/index|Public logs]] |
