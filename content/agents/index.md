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

- [[agents/core/index|Agent Core]]
- [[agents/tools/index|Agent Tools]]
- [[agents/workflows/index|Agent Workflows]]
- [[agents/verification/index|Agent Verification]]

## Routing Guide

- Put model-state-memory-planning ideas in [[agents/core/index|Agent Core]].
- Put tool schemas, side effects, permissions, and output handling in [[agents/tools/index|Agent Tools]].
- Put repeatable use cases such as coding, paper briefs, and wiki maintenance in [[agents/workflows/index|Agent Workflows]].
- Put acceptance criteria, evidence, audits, review, and safety checks in [[agents/verification/index|Agent Verification]].
- Put Korean reader-facing narratives in [[posts/index|Posts]] only after the underlying wiki notes are stable.

## Core Concepts

- [[agents/core/agent-architecture|Agent architecture]]
- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/core/agent-environment|Agent environment]]
- [[agents/core/action-space|Action space]]
- [[agents/core/agent-state|Agent state]]
- [[agents/core/context-engineering|Context engineering]]
- [[concepts/llm/prompting|Prompting]]
- [[concepts/llm/structured-output|Structured output]]
- [[agents/tools/tool-use|Tool use]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/tools/tool-result-handling|Tool result handling]]
- [[agents/core/planning|Planning]]
- [[agents/core/task-decomposition|Task decomposition]]
- [[agents/core/agent-memory|Agent memory]]
- [[agents/core/memory-boundary|Memory boundary]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/acceptance-criteria|Acceptance criteria]]
- [[agents/verification/evidence-ledger|Evidence ledger]]
- [[agents/verification/completion-audit|Completion audit]]
- [[agents/verification/reflection-and-critique|Reflection and critique]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[agents/verification/prompt-injection|Prompt injection]]

## Workflows

- [[agents/workflows/coding-agents|Coding agents]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[agents/workflows/agent-orchestration|Agent orchestration]]
- [[agents/workflows/agent-handoff|Agent handoff]]
- [[agents/workflows/agent-runbook|Agent runbook]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
- [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]]

## Learning and Feedback

- [[concepts/learning/imitation-learning|Imitation learning]]
- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[concepts/learning/reward-modeling|Reward modeling]]
- [[concepts/learning/preference-optimization|Preference optimization]]

## Writing Rules

- Explain workflows with generic examples.
- Keep vendor-specific claims dated or scoped.
- Do not publish private prompts, private repo details, credentials, internal task names, or unreleased work.
- Prefer concrete verification habits over broad productivity claims.

## Related

- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[concepts/llm/hallucination-grounding|Hallucination and grounding]]
- [[concepts/architectures/transformer|Transformer]]
- [[inbox/index|Inbox]]
- [[projects/index|Projects]]
- [[logs/index|Public logs]]
