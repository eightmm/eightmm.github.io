---
title: Agent Operating Contract
tags:
  - agents
  - workflows
  - verification
---

# Agent Operating Contract

An agent operating contract defines what an agent is allowed to do, what evidence it must inspect, when it must stop, and how completion is verified. It turns an open-ended instruction into a bounded workflow.

A compact contract can be written as:

$$
C =
(\mathcal{G}, \mathcal{S}, \mathcal{A}, \mathcal{V}, \mathcal{B})
$$

where $\mathcal{G}$ is the goal, $\mathcal{S}$ is the observable state, $\mathcal{A}$ is the allowed action space, $\mathcal{V}$ is the verification set, and $\mathcal{B}$ is the boundary or stop condition.

## Contract Parts

- Goal: the artifact or decision the agent is trying to produce.
- State: files, logs, rendered pages, issues, run outputs, or external evidence the agent must inspect.
- Action space: tools and edits the agent can perform.
- Verification: commands, reviews, checks, screenshots, or acceptance criteria required before reporting success.
- Boundaries: private information, destructive actions, high-risk changes, and cases requiring human approval.

## Completion Rule

Completion should be evidence-based:

$$
\operatorname{done}(g)
=
\bigwedge_{v\in\mathcal{V}(g)}
\operatorname{pass}(v)
$$

If a required check is skipped, unavailable, or too narrow for the goal, the agent should report the gap instead of claiming completion.

## Public Wiki Use

For a public research blog or LLM Wiki, the operating contract should include:

- No private server details, credentials, private paths, collaborator details, or unpublished results.
- Source-grounded paper metadata.
- Wikilink integrity.
- Build verification.
- Clear distinction between draft, inbox, curated note, and published post.

## Checks

- Is the goal specific enough to verify?
- Are side effects allowed and reversible?
- Are private boundaries explicit?
- Does the verification set match the blast radius?
- Is the final report tied to evidence rather than model confidence?

## Related

- [[agents/core/agent-loop|Agent loop]]
- [[agents/core/action-space|Action space]]
- [[agents/core/planning|Planning]]
- [[agents/verification/acceptance-criteria|Acceptance criteria]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/workflows/agent-runbook|Agent runbook]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
