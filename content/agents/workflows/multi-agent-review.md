---
title: Multi-Agent Review
tags:
  - agents
  - llm
  - review
---

# Multi-Agent Review

Multi-agent review uses more than one agent — often different models or roles — to independently inspect a change before it is accepted. Independent passes catch errors a single agent rationalizes away.

The value comes from conditional independence. If reviewers share the same hidden assumption, agreement is weak evidence:

$$
P(\text{correct}\mid r_1,r_2)
\not\approx
P(\text{correct}\mid r_1)P(\text{correct}\mid r_2)
$$

unless the reviewers inspected the artifact independently and used different failure hypotheses.

## Useful Roles

- Implementer: makes the smallest scoped change.
- Reviewer: searches for bugs, missing checks, and regressions.
- Domain critic: checks scientific, data, or security assumptions.
- Verifier: runs acceptance checks and records evidence.
- Editor: turns rough output into public documentation.

## Practical Checks

- Give each reviewer the diff plus the goal, not a summary that hides detail.
- Prefer adversarial prompts ("try to refute this") over confirmation.
- Treat agreement across independent models as weak evidence, not proof.
- Gate high-risk surfaces — APIs, auth, data pipelines, training, deps — behind review.
- Record verdicts so a decision can be re-examined later.
- Do not treat reviewer consensus as a substitute for running tests or inspecting sources.
- Keep private data out of prompts and review artifacts.

## Public Wiki Use

For a research blog, multi-agent review is most useful for:

- Sanitizing public posts.
- Checking paper-note claims against source metadata.
- Finding broken links or missing definitions.
- Reviewing agent-generated Markdown before publication.
- Separating raw inbox notes from curated wiki pages.

## Related

- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/core/planning|Planning]]
- [[agents/workflows/coding-agents|Coding agents]]
- [[agents/index|Agents]]
