---
title: Paper Candidate Intake
tags:
  - inbox
  - papers
  - workflows
---

# Paper Candidate Intake

Paper candidate intake is the minimum schema for papers collected by discovery agents before they become curated notes. It keeps raw candidates useful without pretending they are verified reviews.

$$
\text{candidate}
=
\text{metadata}
+ \text{route}
+ \text{claim}
+ \text{evidence}
+ \text{next action}
$$

## Minimum Fields

| Field | Required Value |
| --- | --- |
| Source | URL, arXiv, DOI, venue page, project page, or `to verify` |
| Metadata | title, authors, year, venue/status, publication type |
| Why collected | one sentence explaining relevance to AI, Molecular Modeling, Math, Infra, Agents, or Projects |
| Primary route | paper, concept update, benchmark note, project note, post idea, or drop |
| Main axis | architecture, learning method, generative model, molecular modeling, Math, evaluation, systems, agents |
| Candidate claim | one narrow claim, written as paper claim |
| Evidence pointer | figure, table, benchmark, theorem, artifact, or `to verify` |
| Risk | missing source, weak fit, benchmark issue, leakage, private info, hype, duplicate |
| Next action | smallest verification or promotion step |
| Status | `inbox`, `triage`, `reading`, `defer`, or `drop` |

## Candidate Block

Use this block inside daily briefs:

```markdown
### Paper title

- Source: to verify
- Metadata: authors `to verify`; year `to verify`; venue/status `to verify`
- Why collected: to verify
- Primary route: paper / concept update / benchmark note / post idea / drop
- Main axis: architecture / learning method / generative model / molecular modeling / Math / evaluation / systems / agents
- Candidate claim: to verify
- Evidence pointer: to verify
- Related wiki notes:
  - [Coverage matrix](/concepts/coverage-matrix)
- Risk: to verify
- Next action: verify metadata
- Status: inbox
```

## Promotion Decision

| Decision | Use When | Start |
| --- | --- | --- |
| Paper note | source is public, metadata is verified, and claims need durable tracking | [Paper triage](/papers/workflows/paper-triage) |
| Concept update | the reusable idea is more important than the single paper | [Concept update contract](/papers/workflows/concept-update-contract) |
| Benchmark note | the evaluation protocol is the main value | [Benchmark intake](/concepts/data/benchmark-intake) |
| Korean post idea | several wiki notes already form a reader-facing path | [AI-Molecular-Math post intake](/posts/ai-molecular-math-post-intake) |
| Curation queue | promising but missing metadata, relevance, or verification | [Curation queue](/inbox/curation-queue) |
| Drop | duplicate, unverifiable, out of scope, or unsafe for public notes | [Inbox triage](/inbox/inbox-triage) |

## Checks

- Does the candidate have a public source?
- Is missing metadata marked `to verify`?
- Is the claim written narrowly as a paper claim?
- Is the main axis chosen before creating a new page?
- Is there a route to concept update when the idea is reusable?
- Are benchmark, split, metric, and artifact issues flagged before promotion?
- Is the next action concrete enough for another agent to continue?

## Related

- [[inbox/index|Inbox]]
- [[inbox/inbox-triage|Inbox triage]]
- [[inbox/curation-queue|Curation queue]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[papers/workflows/paper-triage|Paper triage]]
- [[papers/workflows/claim-routing|Claim routing]]
- [[papers/workflows/concept-update-contract|Concept update contract]]
- [[concepts/coverage-matrix|Coverage matrix]]
