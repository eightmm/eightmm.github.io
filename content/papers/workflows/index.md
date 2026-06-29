---
title: Paper Workflows
unlisted: true
tags:
  - papers
  - workflows
---

# Paper Workflows

Paper workflow note는 raw paper candidate가 오래 남길 수 있는 공개 노트로 바뀌는 절차를 정의합니다.

Paper는 명시적인 상태를 거쳐 이동해야 합니다.

$$
\text{candidate}
\rightarrow
\text{triaged}
\rightarrow
\text{reading}
\rightarrow
\text{verified or archived}
$$

이 workflow의 목적은 거친 daily brief가 완성된 paper review처럼 보이지 않게 하는 것입니다.

## Scope

- triage decision과 reading status.
- 선별된 paper note의 기본 형태.
- 20-30분짜리 beginner-friendly longform paper review.
- paper bucket routing.
- primary claim axis routing.
- metadata check에서 concept update까지 이어지는 review workflow.
- AI, Computational Biology, Math foundations를 통한 domain intake.
- multi-axis note나 synthesis post로 승격하기 전 readiness check.
- 사람이 읽을 수 있게 sanitize한 뒤 쓰는 agent-assisted paper brief.

## 노트

- [[papers/workflows/paper-triage|Paper triage]]
- [[papers/workflows/claim-routing|Claim routing]]
- [[papers/workflows/paper-to-wiki-extraction|Paper to wiki extraction]]
- [[papers/workflows/concept-update-contract|Concept update contract]]
- [[papers/workflows/ai-molecular-math-readiness-gate|AI Computational Biology Math readiness gate]]
- [[papers/workflows/reading-status|Reading status]]
- [[papers/workflows/paper-note-format|Paper note format]]
- [[papers/workflows/longform-paper-review-guide|Longform paper review guide]]
- [[papers/workflows/ai-molecular-math-paper-template|AI Computational Biology Math paper template]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[ai/paper-intake|AI paper intake]]
- [[molecular-modeling/paper-intake|Computational Biology paper intake]]
- [[math/formula-intake|Formula intake]]

## 확인할 것

- paper source가 public이고 metadata가 검증되었는가?
- note status가 보이고 정직하게 표시되는가?
- output이 paper note, concept update, inbox item, synthesis post, archive decision 중 무엇인가?
- 빠진 claim, metric, author, artifact를 지어내지 않고 `to verify`로 표시했는가?
- AI/Computational Biology/Math paper에는 [[papers/workflows/ai-molecular-math-paper-template|AI Computational Biology Math paper template]]을 썼는가?
- 공개 blog-style paper review에는 [[papers/workflows/longform-paper-review-guide|Longform paper review guide]]를 썼는가?
- primary claim axis가 [[papers/workflows/claim-routing|Claim routing]]에 따라 기록되었는가?
- paper claim이 [[papers/workflows/paper-to-wiki-extraction|Paper to wiki extraction]]에 따라 object, representation, method, formula, data, evidence, artifact update로 분해되었는가?
- 재사용 가능한 definition, formula, contract, evidence boundary가 [[papers/workflows/concept-update-contract|Concept update contract]]로 추출되었는가?
- multi-axis candidate가 승격 전에 [[papers/workflows/ai-molecular-math-readiness-gate|AI Computational Biology Math readiness gate]]를 통과했는가?
- workflow가 paper summary만 쌓지 않고 관련 concept page를 업데이트하는가?

## 새 노트 위치

- process와 status note는 여기에 둡니다.
- claim, evidence, benchmark, comparison note는 [[papers/analysis/index|Paper analysis]]에 둡니다.
- artifact와 reproduction note는 [[papers/reproducibility/index|Paper reproducibility]]에 둡니다.
- raw candidate는 [[inbox/index|Inbox]]에 남깁니다.

## Related

- [[inbox/index|Inbox]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[papers/analysis/index|Paper analysis]]
- [[papers/reproducibility/index|Paper reproducibility]]
- [[ai/index|AI]]
- [[molecular-modeling/index|Computational Biology]]
- [[math/index|Math]]
