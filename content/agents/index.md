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
| [Agent Core](/agents/core) | architecture, loop, state, memory, planning |
| [Agent Tools](/agents/tools) | tool contracts, side effects, result handling |
| [Agent Workflows](/agents/workflows) | coding, paper briefs, orchestration, handoff |
| [Agent Verification](/agents/verification) | acceptance criteria, evidence, audits, evaluation |

## Routing Guide

| If the note is about... | Put it in |
| --- | --- |
| model state, memory, planning, context | [Agent Core](/agents/core) |
| schemas, side effects, permissions, outputs | [Agent Tools](/agents/tools) |
| repeatable use cases and handoffs | [Agent Workflows](/agents/workflows) |
| acceptance criteria, evidence, audits, safety checks | [Agent Verification](/agents/verification) |
| broad reader-facing stories | [Posts](/posts) |

## Core Concepts

| Group | Notes |
| --- | --- |
| Core model | [Agent architecture](/agents/core/agent-architecture), [Agent operating contract](/agents/core/agent-operating-contract), [Agent loop](/agents/core/agent-loop) |
| Environment and state | [Agent environment](/agents/core/agent-environment), [Action space](/agents/core/action-space), [Agent state](/agents/core/agent-state) |
| Context and memory | [Context engineering](/agents/core/context-engineering), [Agent memory](/agents/core/agent-memory), [Memory boundary](/agents/core/memory-boundary) |
| Planning | [Planning](/agents/core/planning), [Task decomposition](/agents/core/task-decomposition) |
| LLM interface | [Prompting](/concepts/llm/prompting), [Structured output](/concepts/llm/structured-output) |
| Tools | [Tool use](/agents/tools/tool-use), [Tool contract](/agents/tools/tool-contract), [Tool result handling](/agents/tools/tool-result-handling) |
| Verification | [Verification loop](/agents/verification/verification-loop), [Acceptance criteria](/agents/verification/acceptance-criteria), [Evidence ledger](/agents/verification/evidence-ledger) |
| Review and safety | [Completion audit](/agents/verification/completion-audit), [Reflection and critique](/agents/verification/reflection-and-critique), [Agent evaluation](/agents/verification/agent-evaluation), [Human in the loop](/agents/verification/human-in-the-loop), [Prompt injection](/agents/verification/prompt-injection) |

## Workflows

| Workflow | Use For |
| --- | --- |
| [Coding agents](/agents/workflows/coding-agents) | implementation and repository work |
| [Paper brief workflow](/agents/workflows/paper-brief-workflow) | paper intake and synthesis |
| [Agent orchestration](/agents/workflows/agent-orchestration) | multiple agents or tools |
| [Agent handoff](/agents/workflows/agent-handoff) | passing state between workers |
| [Agent runbook](/agents/workflows/agent-runbook) | repeatable operating procedure |
| [Multi-agent review](/agents/workflows/multi-agent-review) | independent verification |
| [LLM Wiki](/agents/workflows/llm-wiki) | knowledge-base maintenance |
| [Content promotion workflow](/agents/workflows/content-promotion-workflow) | inbox to note/post/project promotion |
| [Paper brief agent pipeline](/projects/paper-brief-agent-pipeline) | project-level paper workflow |

## Learning and Feedback

| Method | Link |
| --- | --- |
| Imitation | [Imitation learning](/concepts/learning/imitation-learning) |
| Reinforcement | [Reinforcement learning](/concepts/learning/reinforcement-learning) |
| Reward modeling | [Reward modeling](/concepts/learning/reward-modeling) |
| Preference objective | [Preference optimization](/concepts/learning/preference-optimization) |

## Writing Rules

- Explain workflows with generic examples.
- Keep vendor-specific claims dated or scoped.
- Do not publish private prompts, private repo details, credentials, internal task names, or unreleased work.
- Prefer concrete verification habits over broad productivity claims.

## Related

| Area | Link |
| --- | --- |
| Knowledge workflow | [LLM Wiki](/agents/workflows/llm-wiki), [Inbox](/inbox) |
| LLM behavior | [Hallucination and grounding](/concepts/llm/hallucination-grounding), [Transformer](/concepts/architectures/transformer) |
| Public work records | [Projects](/projects), [Public logs](/logs) |
