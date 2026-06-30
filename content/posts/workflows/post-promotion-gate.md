---
title: Post Promotion Gate
aliases:
  - posts/post-promotion-gate
tags:
  - posts
  - workflow
  - publishing
---

# Post Promotion Gate

Post promotion gate는 wiki note cluster가 Korean reader-facing post가 될 준비가 되었는지 판단합니다. Wiki가 준비되기 전에 얕은 post를 공개하거나, 이미 mature한 topic을 disconnected note로 묻어 두는 문제를 막기 위한 gate입니다.

$$
\text{post-ready}
=
\text{question}
\land
\text{wiki bundle}
\land
\text{evidence boundary}
\land
\text{public-safe}
\land
\text{next path}
$$

## Promotion Fields

| Field | Pass when |
| --- | --- |
| Reader question | post가 clear question 하나에 답합니다 |
| Primary axis | AI, Computational Biology, Math, Paper cluster, Project, Infra, Agents 중 하나가 중심입니다 |
| Topic map | broad post가 loose link list가 아니라 explicit map contract를 가집니다 |
| Wiki bundle | required support note가 [Wiki bundle checklist](/posts/workflows/wiki-bundle-checklist)를 통과합니다 |
| Minimum formula | 필요한 equation이 symbol과 함께 포함되거나 Math note로 연결됩니다 |
| Evidence boundary | paper, benchmark, split, metric, baseline, leakage, uncertainty claim의 scope가 정해져 있습니다 |
| Personal angle | post가 reading order, interpretation, practical judgment를 더합니다 |
| Public boundary | private infrastructure, unpublished result, collaborator detail, internal task가 없습니다 |
| Next path | reader가 이어서 볼 3-7개의 useful link를 가집니다 |

## Wiki Bundle

| Bundle type | Minimum links |
| --- | --- |
| AI method post | architecture, learning method, objective, evaluation risk |
| Computational Biology post | entity/object, representation, preprocessing/split, evaluation risk |
| Math-heavy post | formula intake, explanation ladder, objective-metric link, evaluation math |
| Paper-cluster post | paper notes, claim/evidence table, benchmark contract, concept updates |
| Project post | project note, design decision, verification, related concepts |
| Infra post | public runbook, failure taxonomy, reproducibility or systems concept |
| Agent workflow post | agent workflow, verification loop, handoff or memory boundary |

## Decision Table

| Candidate state | Destination |
| --- | --- |
| definition 또는 equation 하나 | concept note |
| paper claim 하나 | paper note |
| unverified source 또는 unclear route | inbox 또는 curation queue |
| paper에서 나온 reusable idea | concept update |
| reader question 하나로 묶이는 여러 note | Korean post |
| implementation narrative | project note |
| public operational lesson | infra 또는 public log |

## Draft Contract

Drafting 전에 아래를 채웁니다.

```yaml
reader_question: to verify
primary_axis: to verify
wiki_bundle:
  - to verify
minimum_formula: not applicable
evidence_boundary: to verify
paper_sources: not applicable
public_boundary: to verify
next_path:
  - to verify
status: draft
```

## Stop Conditions

- post가 term 하나를 정의하는 데 대부분을 쓴다.
- post가 이해되려면 private context가 필요하다.
- reusable wiki link가 없다.
- broad map인데 [[concepts/topic-map-contract|Topic map contract]]를 통과하지 않는다.
- metadata 또는 evidence가 아직 `to verify`인 paper claim에 의존한다.
- Concepts, Papers, Math, Infra에 있어야 할 긴 definition을 반복한다.
- next reading path가 없다.

## Related

- [[posts/essays/wiki-to-post-workflow|Wiki to post workflow]]
- [[posts/workflows/ai-molecular-math-post-intake|AI Computational Biology Math post intake]]
- [[posts/workflows/wiki-bundle-checklist|Wiki bundle checklist]]
- [[posts/workflows/synthesis-post-template|Synthesis post template]]
- [[posts/workflows/blog-writing-guide|Blog writing guide]]
- [[concepts/topic-map-contract|Topic map contract]]
- [[concepts/coverage-matrix|Coverage matrix]]
- [[papers/workflows/ai-molecular-math-readiness-gate|AI Computational Biology Math readiness gate]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
