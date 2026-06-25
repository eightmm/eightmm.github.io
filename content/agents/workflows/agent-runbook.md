---
title: Agent Runbook
tags:
  - agents
  - workflows
  - runbook
---

# Agent Runbook

An agent runbook is a reusable operating procedure for a recurring agent workflow. It turns vague repeated work into a checklist with inputs, steps, checks, and public-output rules.

A runbook can be viewed as:

$$
R
=
(\mathcal{I}, \mathcal{S}, \mathcal{V}, \mathcal{O})
$$

where $\mathcal{I}$ is input contract, $\mathcal{S}$ is ordered steps, $\mathcal{V}$ is verification, and $\mathcal{O}$ is output contract.

## Runbook Sections

- Purpose: what workflow this covers.
- Inputs: required files, prompts, tickets, papers, or data.
- Preconditions: branch, environment, privacy boundary, and approval needs.
- Steps: ordered actions with stop conditions.
- Verification: commands, review gates, and evidence.
- Output: summary, changed files, open questions, and next action.
- Sanitization: what must not be published.

## Useful Runbooks

- Daily paper brief ingestion.
- Blog/wiki note curation.
- Coding agent patch admission.
- Multi-agent review.
- HPC failure summary.
- Public incident or experiment note cleanup.

## Checks

- Does the runbook prevent repeated ambiguity?
- Does it include a verification step after edits?
- Does it state what not to publish?
- Can a different agent follow it without private context?
- Does it produce durable wiki improvements rather than one-off chat output?

## Related

- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[agents/workflows/agent-handoff|Agent handoff]]
- [[agents/verification/verification-loop|Verification loop]]
- [[posts/blog-writing-guide|블로그 글 작성 가이드]]
