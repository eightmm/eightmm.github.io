---
title: Agent Memory
tags:
  - agents
  - llm
  - memory
---

# Agent Memory

Agent memory는 agent가 single context window를 넘어 정보를 유지하는 방식입니다. Scratch note, durable fact, prior decision, retrieved document가 여기에 포함됩니다. Memory가 없으면 매 turn이 blind start가 되고, 너무 많으면 context가 noise로 찹니다.

Memory는 하나의 물건이 아닙니다. 유용한 분해는 아래와 같습니다.

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

여기서 working memory는 task-local state, episodic memory는 past run 기록, semantic memory는 reusable concept 또는 preference, external memory는 file, doc, database, search에서 retrieved된 정보입니다.

## Retrieval View

Agent는 보통 모든 memory를 prompt에 넣을 수 없습니다. 대신 작은 subset을 retrieve합니다.

$$
R(q,M)
=
\operatorname{topk}_{m\in M}
\operatorname{score}(q,m)
$$

여기서 $q$는 현재 task/query이고 $m$은 memory record입니다. Scoring function은 lexical matching, embedding, metadata filter, recency, source authority, learned ranker를 사용할 수 있습니다.

좋은 memory는 selective합니다. 올바른 질문은 “agent가 이것을 기억할 수 있는가?”가 아니라 “이 task에서 이 fact가 retrieve되어야 하는가?”입니다.

## Memory record

Durable memory record에는 보통 아래 항목이 필요합니다.

- Claim: fact 또는 reusable rule.
- Scope: 적용되는 범위.
- Source: 어디서 온 정보인지.
- Date or version: 언제 true였는지.
- Sensitivity: public, private, secret, prohibited 중 무엇인지.
- Revalidation rule: 언제 다시 확인해야 하는지.

High-impact action에서는 recalled memory를 hypothesis로 취급해야 합니다.

$$
\operatorname{act}(m)
\Rightarrow
\operatorname{verify}(m)
$$

Memory가 오래됐거나 결과가 클수록 더 강한 verification이 필요합니다.

## Failure mode

- Stale memory: 오래된 fact를 current fact처럼 취급합니다.
- Over-retrieval: irrelevant note가 task context를 밀어냅니다.
- Under-retrieval: 중요한 constraint를 놓칩니다.
- Contaminated memory: injected content나 잘못된 content가 persist됩니다.
- Privacy leak: private detail이 저장되거나 public workflow로 recall됩니다.
- Preference drift: agent가 좁은 user preference를 과도하게 일반화합니다.

## 실전 check

- short-term working state와 durable, reusable fact를 분리합니다.
- recall이 selective하게 유지되도록 record 하나에는 fact 하나와 짧은 description을 둡니다.
- 모든 것을 context에 넣기보다 필요할 때 retrieve합니다.
- recalled fact로 행동하기 전에 여전히 맞는지 verify합니다.
- secret, private path, unpublished result를 persist하지 않습니다.
- memory가 user instruction, project convention, observed fact, inferred preference 중 무엇인지 추적합니다.
- software version, URL, schedule, people, policy, infrastructure에 묶인 memory는 expire 또는 revalidate합니다.

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
