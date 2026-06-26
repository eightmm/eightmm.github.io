---
title: Agent Memory
tags:
  - agents
  - llm
  - memory
---

# Agent Memory

Agent memory is how an agent carries information beyond a single context window — scratch notes, durable facts, prior decisions, and retrieved documents. Without it, every turn starts blind; with too much, context fills with noise.

Memory is not one thing. A useful decomposition is:

$$
M
=
M_{\mathrm{working}}
\cup
M_{\mathrm{episodic}}
\cup
M_{\mathrm{semantic}}
\cup
M_{\mathrm{external}}
$$

where working memory is task-local state, episodic memory records past runs, semantic memory stores reusable concepts or preferences, and external memory is retrieved from files, docs, databases, or search.

## Retrieval View

An agent usually cannot load all memory into the prompt. It retrieves a small subset:

$$
R(q,M)
=
\operatorname{topk}_{m\in M}
\operatorname{score}(q,m)
$$

where $q$ is the current task/query and $m$ is a memory record. The scoring function can use lexical matching, embeddings, metadata filters, recency, source authority, or a learned ranker.

Good memory is selective. The right question is not "can the agent remember this?" but "should this fact be retrieved for this task?"

## Memory Record

A durable memory record should usually include:

- Claim: the fact or reusable rule.
- Scope: where it applies.
- Source: where it came from.
- Date or version: when it was true.
- Sensitivity: public, private, secret, or prohibited.
- Revalidation rule: when to check again.

For high-impact actions, recalled memory should be treated as a hypothesis:

$$
\operatorname{act}(m)
\Rightarrow
\operatorname{verify}(m)
$$

The older or more consequential the memory is, the stronger the verification should be.

## Failure Modes

- Stale memory: old facts are treated as current.
- Over-retrieval: irrelevant notes crowd out task context.
- Under-retrieval: important constraints are missed.
- Contaminated memory: injected or wrong content is persisted.
- Privacy leak: private details are stored or recalled into a public workflow.
- Preference drift: the agent overgeneralizes a narrow user preference.

## Practical Checks

- Separate short-term working state from durable, reusable facts.
- Store one fact per record with a short description so recall stays selective.
- Prefer retrieval on demand over loading everything into context.
- Verify a recalled fact still holds before acting on it.
- Avoid persisting secrets, private paths, or unpublished results.
- Track whether a memory is user instruction, project convention, observed fact, or inferred preference.
- Expire or revalidate memories tied to software versions, URLs, schedules, people, policies, or infrastructure.

## Related

- [[agents/core/memory-boundary|Memory boundary]]
- [[agents/verification/evidence-ledger|Evidence ledger]]
- [[agents/verification/prompt-injection|Prompt injection]]
- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[concepts/llm/context-window|Context window]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/planning|Planning]]
- [[agents/tools/tool-use|Tool use]]
- [[agents/index|Agents]]
- [[concepts/evaluation/index|Evaluation]]
