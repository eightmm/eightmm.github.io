---
title: AI Computational Biology Math 포스트 인테이크
aliases:
  - posts/ai-bio-math-post-intake
  - posts/ai-computational-biology-math-post-intake
tags:
  - posts
  - writing
  - ai
  - molecular-modeling
  - math
---

# AI Computational Biology Math 포스트 인테이크

AI, computational biology, Math가 함께 나오는 글은 처음부터 모든 것을 설명하려고 하면 흐려집니다. 한글 post는 독자가 읽을 경로와 판단 기준을 잡아 주고, 정확한 정의와 수식은 wiki note로 넘기는 것이 좋습니다.

$$
\text{post}
=
\text{question}
+ \text{map}
+ \text{minimum concepts}
+ \text{evidence boundary}
+ \text{next links}
$$

## 먼저 정할 것

| Field | Question | Route |
| --- | --- | --- |
| Reader question | 이 글이 답하는 한 문장 질문은 무엇인가? | [Blog writing guide](/posts/blog-writing-guide) |
| Main axis | AI 방법, computational biology 대상, Math 수식, paper claim 중 무엇이 중심인가? | [AI](/ai), [Computational Biology](/molecular-modeling), [Math](/math) |
| Cross-axis contract | object, representation, model, objective, evidence가 모두 분리됐는가? | [AI Computational Biology Math contract](/concepts/ai-computational-biology-math-contract) |
| Required wiki notes | 글 안에서 반복하지 않고 링크할 개념은 무엇인가? | [Concepts](/concepts) |
| Formula need | 독자 이해에 필요한 최소 수식은 무엇인가? | [Formula intake](/math/formula-intake) |
| Benchmark boundary | 성능 claim을 어디까지 믿을 수 있는가? | [Benchmark intake](/concepts/data/benchmark-intake) |
| Coverage check | 연결해야 할 object, data, method, formula, evidence note가 있는가? | [Coverage matrix](/concepts/coverage-matrix) |
| Post promotion | wiki bundle, reader question, evidence boundary, next path가 충분한가? | [Post promotion gate](/posts/post-promotion-gate) |
| Readiness gate | route, representation, objective, evidence, public boundary가 통과됐는가? | [AI Computational Biology Math readiness gate](/papers/workflows/ai-molecular-math-readiness-gate) |
| Public boundary | 공개하면 안 되는 내부 정보가 섞여 있지 않은가? | [Publishing gate](/inbox/publishing-gate) |

## 중심축별 글 모양

| Main Axis | Post Should Explain | Wiki Should Hold |
| --- | --- | --- |
| AI method | architecture, learning signal, objective, evaluation risk를 읽는 순서 | full definitions, equations, variants, checklists |
| Computational biology object | protein, molecule, ligand, pocket, conformer, complex가 왜 다른 문제인지 | entity contracts, preprocessing, split and leakage details |
| Math formula | 수식이 어떤 object, distribution, objective, metric을 뜻하는지 | derivations, canonical formulas, symbol tables |
| Paper cluster | 여러 논문이 같은 질문에 어떻게 다른 답을 주는지 | individual paper claims, evidence tables, benchmark cards |
| Project or workflow | 어떤 문제를 해결하려 했고 어떤 설계 판단을 했는지 | implementation details, verification logs, artifact contracts |

## 글 안에 넣을 것

- 문제의식: 왜 지금 이 주제를 읽어야 하는가.
- 큰 지도: entity, modality, architecture, learning method, evaluation 중 어떤 축으로 볼 것인가.
- 최소 수식: 독자가 claim을 이해하는 데 필요한 식만 넣고, 모든 기호를 설명합니다.
- 판단 기준: split, metric, leakage, baseline, uncertainty, benchmark boundary를 짧게 둡니다.
- 링크 경로: AI, Computational Biology, Math, Papers, Concepts로 이어지는 다음 경로를 둡니다.

## 글 밖으로 뺄 것

- 용어 하나의 긴 정의: [[concepts/index|Concepts]]로 분리합니다.
- 논문 하나의 세부 claim: [[papers/index|Papers]]로 분리합니다.
- benchmark 세부 항목: [[concepts/data/benchmark-intake|Benchmark intake]]나 [[papers/analysis/benchmark-card|Benchmark card]]로 분리합니다.
- 수식 전개와 기호표: [[math/formula-intake|Formula intake]]나 관련 math note로 분리합니다.
- 구현/운영 기록: [[projects/index|Projects]]나 [[infra/index|Infra]]로 분리합니다.

## 추천 구조

실제 초안은 [[posts/synthesis-post-template|Synthesis post template]]을 복사하지 말고 구조만 따라 씁니다.

```markdown
---
title: 글 제목
date: YYYY-MM-DD
tags:
  - posts
---

# 글 제목

## 왜 이 질문인가

독자가 얻을 관점과 문제의식.

## 큰 지도

AI / Computational Biology / Math 중 어떤 축으로 나눠 볼지 설명.

## 최소 개념

필요한 수식이나 정의를 짧게 설명하고 wiki note로 연결.

## 논문이나 benchmark를 볼 때

claim, split, metric, leakage, baseline, uncertainty를 점검.

## 다음에 볼 노트

관련 concept, paper, project, infra, agent workflow 링크.
```

## Checks

- 글 하나가 하나의 질문에 답하는가?
- AI 방법론, computational biology 대상, Math 수식 중 중심축이 분명한가?
- 최소 수식과 symbol 설명이 들어갔는가?
- 세부 정의를 post에 반복하지 않고 wiki note로 넘겼는가?
- paper나 benchmark claim은 evidence boundary를 갖는가?
- multi-axis 주제는 readiness gate를 통과했는가?
- post로 승격할 만큼 wiki bundle과 next path가 충분한가?
- 내부 정보, 미공개 결과, 서버 정보, collaborator 정보가 빠졌는가?

## Related

- [[posts/blog-writing-guide|블로그 글 작성 가이드]]
- [[posts/synthesis-post-template|Synthesis post template]]
- [[posts/post-promotion-gate|Post promotion gate]]
- [[posts/wiki-to-post-workflow|Wiki에서 Post로 승격하는 방식]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
- [[concepts/ai-computational-biology-math-contract|AI Computational Biology Math contract]]
- [[ai/paper-intake|AI paper intake]]
- [[molecular-modeling/paper-intake|Molecular modeling paper intake]]
- [[math/formula-intake|Formula intake]]
- [[concepts/data/benchmark-intake|Benchmark intake]]
- [[concepts/coverage-matrix|Coverage matrix]]
- [[papers/workflows/ai-molecular-math-readiness-gate|AI Computational Biology Math readiness gate]]
