---
title: Research Methodology
tags:
  - research
  - methodology
  - experiments
---

# Research Methodology

Research methodology note는 아이디어를 testable claim으로 바꾸고, 작은 experiment를 실행하고, result를 해석하고, public note를 정직하게 유지하는 방법을 정의합니다.

핵심 loop는 아래와 같습니다.

$$
\text{question}
\rightarrow \text{hypothesis}
\rightarrow \text{prediction}
\rightarrow \text{experiment}
\rightarrow \text{result}
\rightarrow \text{revision}
$$

## Research Contract

Research note는 아이디어, 실험, 결과, 공개 글을 섞지 않고 아래 계약으로 나눕니다.

$$
R
=
(q,\ h,\ p,\ e,\ a,\ c,\ l)
$$

| Part | Meaning | Write |
| --- | --- | --- |
| $q$ | research question | 무엇을 알고 싶은가 |
| $h$ | hypothesis | 어떤 mechanism 또는 explanation을 가정하는가 |
| $p$ | prediction | hypothesis가 맞다면 무엇이 관찰되어야 하는가 |
| $e$ | experiment or evidence source | 어떤 최소 실험, paper, proof, benchmark를 볼 것인가 |
| $a$ | artifact | code, run record, table, figure, note, dataset snapshot |
| $c$ | claim | evidence가 뒷받침하는 좁은 문장 |
| $l$ | limitation | 무엇을 아직 증명하지 못했는가 |

좋은 public research note는 결론보다 먼저 claim의 범위를 좁힙니다.

$$
\text{evidence}
\Rightarrow
\text{claim within scope}
\not\Rightarrow
\text{universal conclusion}
$$

## Public Research Flow

Private working notes can be messy. Public notes should be promoted only after the claim and boundary are clear.

| Stage | Private working form | Public form |
| --- | --- | --- |
| Idea | rough question, speculation, possible experiments | [[concepts/research-methodology/research-question|Research question]] |
| Hypothesis | tentative explanation | [[concepts/research-methodology/hypothesis|Hypothesis]] with falsifiable prediction |
| Experiment | run command, config, debug logs | [[concepts/research-methodology/experiment-design|Experiment design]] and public-safe method |
| Result | raw metrics, failures, screenshots, logs | [[concepts/research-methodology/result-interpretation|Result interpretation]] |
| Evidence | artifact paths, paper tables, run IDs | [[concepts/research-methodology/claim-evidence-record|Claim evidence record]] |
| Decision | continue, stop, revise, publish | [[concepts/research-methodology/decision-record|Decision record]] |

Do not publish private paths, server names, account names, collaborator details, internal task names, or unpublished experimental results. Convert them into reusable method, limitation, or lesson.

## 핵심 노트

- [[concepts/research-methodology/research-question|Research question]]
- [[concepts/research-methodology/hypothesis|Hypothesis]]
- [[concepts/research-methodology/experiment-design|Experiment design]]
- [[concepts/research-methodology/minimum-viable-experiment|Minimum viable experiment]]
- [[concepts/systems/experiment-lifecycle|Experiment lifecycle]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
- [[concepts/research-methodology/claim-evidence-record|Claim evidence record]]
- [[concepts/research-methodology/threat-to-validity|Threat to validity]]
- [[concepts/research-methodology/research-log|Research log]]
- [[concepts/research-methodology/experiment-ledger|Experiment ledger]]
- [[concepts/research-methodology/negative-result|Negative result]]
- [[concepts/research-methodology/literature-synthesis|Literature synthesis]]
- [[concepts/research-methodology/decision-record|Decision record]]

## Where It Goes

| Material | Put it in | Rule |
| --- | --- | --- |
| reusable research method | [[concepts/research-methodology/index|Research Methodology]] | general principle, checklist, failure mode |
| tentative research direction | [[research/index|Research]] | idea, hypothesis, open question, direction |
| implemented artifact or pipeline | [[projects/index|Projects]] | code, workflow, tool, report generator, deployment |
| specific paper reading | [[papers/index|Papers]] | paper-specific claim, method, evidence, limitation |
| reflective or dated public writing | [[posts/index|Posts]] | narrative, roadmap, update, Korean-facing blog post |
| system/run reproducibility | [[ai/systems|AI Systems]], [[infra/reproducibility/run-record|Run record]] | artifact boundary and verification |

## Claim Strength Ladder

Not every note needs a full experiment, but the strength of the language should match the evidence.

| Evidence | Safe wording |
| --- | --- |
| intuition only | possible direction, question, motivation |
| one paper or source | reported by this paper, under this benchmark |
| reproduced small example | observed in a small check, not yet general |
| controlled experiment | supports this claim under this setup |
| multiple seeds/splits/baselines | stronger evidence, still scoped to task and data |
| deployment or external validation | operational evidence, still tied to environment and usage |

## Minimum Experiment Shape

Before spending time on a broad run, reduce the experiment to the smallest version that can change the next decision.

| Field | Question |
| --- | --- |
| Decision | What action will change if this result is positive or negative? |
| Baseline | What is the simplest comparison that makes the result meaningful? |
| Variable | What one thing is being changed? |
| Metric | Which metric answers the research question? |
| Failure mode | What result would falsify or weaken the hypothesis? |
| Artifact | What must be saved for later review? |
| Public boundary | What must be removed or generalized before publication? |

## Common Failure Modes

| Failure | Result |
| --- | --- |
| starting from a model name instead of a question | unclear claim and weak comparison |
| running a large experiment before a falsifiable prediction | expensive result that does not guide a decision |
| changing multiple variables at once | impossible attribution |
| omitting negative or inconclusive results | repeated mistakes and biased public notes |
| publishing exact private operational detail | security or collaboration risk |
| copying paper claims without synthesis | paper note does not become reusable knowledge |

## 확인할 것

- 어떤 observation이 conclusion을 바꿀 수 있는가?
- hypothesis가 falsifiable한가?
- baseline 또는 ablation이 있는가?
- 첫 experiment가 decision을 바꿀 수 있는 가장 작은 실험인가?
- 한 번에 하나의 variable만 바꾸는가?
- 어떤 threat to validity가 conclusion을 약하게 만들 수 있는가?
- 성공, 실패, 예상 밖 결과와 상관없이 result를 기록하는가?
- run이 artifact, metric, reproducible record와 연결되어 있는가?
- public claim이 claim evidence record와 연결되어 있는가?
- negative 또는 inconclusive result를 reusable lesson으로 보존하는가?
- paper claim을 복사 요약하지 않고 research question으로 synthesize하는가?
- public note가 private implementation detail과 unpublished result에서 분리되어 있는가?

## Related

- [[research/index|Research]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/research-methodology/claim-evidence-record|Claim evidence record]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[infra/reproducibility/run-record|Reproducible run record]]
