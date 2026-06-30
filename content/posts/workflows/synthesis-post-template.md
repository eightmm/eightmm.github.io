---
title: Synthesis Post Template
aliases:
  - posts/synthesis-post-template
unlisted: true
tags:
  - posts
  - writing
  - synthesis
---

# Synthesis Post Template

AI, computational biology, Math가 섞인 글은 한 번에 많은 개념을 설명하려고 하면 흐름이 흐려집니다. 먼저 한 문장 질문을 정하고, 본문은 독자가 읽을 지도와 판단 기준을 얻도록 씁니다.

$$
\text{synthesis post}
=
\text{reader question}
+ \text{axis map}
+ \text{minimum formulas}
+ \text{evidence boundary}
+ \text{next path}
$$

## Frontmatter

```markdown
---
title: 글 제목
date: YYYY-MM-DD
tags:
  - posts
---
```

## Draft Skeleton

```markdown
# 글 제목

## 왜 이 질문인가

이 글이 답하려는 한 문장 질문.

## 먼저 볼 지도

| 축 | 이 글에서 볼 것 | 더 볼 곳 |
| --- | --- | --- |
| AI | architecture, learning method, objective, evaluation 중 무엇이 중심인가 | [AI](/ai) |
| Computational biology | molecule, protein, ligand, pocket, conformer, complex 중 무엇이 중심인가 | [Computational Biology](/molecular-modeling) |
| Representation | raw object가 model input으로 어떻게 바뀌는가 | [Representation contract](/concepts/modalities/representation-contract) |
| Coordinates | 좌표계, symmetry, pose/RMSD claim이 필요한가 | [Coordinate modeling contract](/concepts/geometric-deep-learning/coordinate-modeling-contract) |
| Math | 어떤 수식, distribution, metric, optimization target이 필요한가 | [Math](/math) |
| Formula depth | post 안에 둘 수식과 wiki note로 넘길 수식을 어떻게 나눌 것인가 | [Formula explanation ladder](/math/formula-explanation-ladder) |
| Evidence | 어떤 benchmark, split, metric, baseline을 믿을 수 있는가 | [Coverage matrix](/concepts/coverage-matrix) |
| Objective | 학습 loss와 reported metric이 같은 claim을 지지하는가 | [Objective-metric alignment](/concepts/machine-learning/objective-metric-alignment) |
| Cross-axis contract | object, representation, model, objective, evidence가 분리됐는가 | [AI Computational Biology Math contract](/concepts/ai-computational-biology-math-contract) |
| Wiki bundle | post를 받쳐 줄 object, method, formula, evidence note가 있는가 | [Wiki bundle checklist](/posts/workflows/wiki-bundle-checklist) |
| Readiness | route, representation, objective, evidence, public boundary가 통과됐는가 | [AI Computational Biology Math readiness gate](/papers/workflows/ai-molecular-math-readiness-gate) |

## 핵심 개념

긴 정의는 반복하지 않고 wiki note로 연결한다.

## 필요한 최소 수식

독자가 claim을 이해하는 데 필요한 수식만 둔다. 더 긴 derivation이나 benchmark claim contract는 wiki note로 넘긴다.

$$
\mathcal{J}(\theta)
=
\mathbb{E}_{u\sim q(u)}
\left[
\ell_\theta(u)
\right]
$$

- $u$: sampled unit
- $q(u)$: sampled distribution
- $\ell_\theta(u)$: loss, score, residual, or metric term
- $\theta$: optimized parameter

## 논문이나 benchmark를 볼 때

| 확인할 것 | 질문 |
| --- | --- |
| Claim | 논문이 실제로 주장하는 것은 무엇인가? |
| Representation | 모델이 실제로 본 입력은 무엇인가? |
| Coordinates | frame, atom/residue mapping, symmetry correction이 명확한가? |
| Split | 어떤 단위가 train/test 사이에서 분리되는가? |
| Metric | metric이 task utility를 직접 반영하는가? |
| Objective | training objective와 selection/test metric이 어긋나지 않는가? |
| Baseline | 비교 대상이 claim에 충분한가? |
| Leakage | 평가 시점에 쓰면 안 되는 정보가 들어갔는가? |
| Artifact | code, data, split, weight, config가 공개되어 있는가? |

결과를 과장해서 해석하기 쉬운 경우 [Claim-evidence boundary](/concepts/evaluation/claim-evidence-boundary)를 같이 봅니다.

## 내 관점

공개 가능한 범위에서 해석, 의문, 다음에 볼 질문을 적는다.

## 다음에 볼 노트

- 관련 AI note
- 관련 computational biology note
- 관련 Math note
- 관련 paper note
- 관련 project 또는 infra note
```

## Required Checks

| Check | Pass when |
| --- | --- |
| Reader question | 글이 한 문장 질문에 답한다 |
| Primary axis | AI, computational biology, Math, paper cluster, project 중 중심축이 분명하다 |
| Claim routing | multi-axis 주제는 [Claim routing](/papers/workflows/claim-routing)을 통과했다 |
| Wiki bundle | [Wiki bundle checklist](/posts/workflows/wiki-bundle-checklist)를 통과했다 |
| Readiness gate | 승격 전 [AI Computational Biology Math readiness gate](/papers/workflows/ai-molecular-math-readiness-gate)를 통과했다 |
| Formula | 필요한 수식과 모든 symbol 설명이 있다 |
| Formula depth | post는 level 1-2, paper/wiki evidence는 level 3 이상으로 분리했다 |
| Evidence boundary | split, metric, baseline, leakage, uncertainty 중 필요한 항목이 있다 |
| Link path | 다음에 읽을 wiki note가 충분히 연결되어 있다 |
| Public boundary | private infrastructure, internal task, unpublished result, collaborator detail이 없다 |

## When Not To Write A Post

- 용어 하나의 정의만 필요하면 concept note로 둡니다.
- 논문 하나의 claim만 다루면 paper note로 둡니다.
- 구현 기록이 중심이면 project note나 infra note로 둡니다.
- 검증되지 않은 daily candidate면 inbox에 둡니다.
- 연결된 wiki note가 거의 없으면 먼저 wiki note를 보강합니다.

## Related

- [[posts/workflows/ai-molecular-math-post-intake|AI Computational Biology Math post intake]]
- [[posts/workflows/wiki-bundle-checklist|Wiki bundle checklist]]
- [[posts/essays/wiki-to-post-workflow|Wiki to post workflow]]
- [[posts/workflows/blog-writing-guide|Blog writing guide]]
- [[papers/workflows/claim-routing|Claim routing]]
- [[concepts/ai-computational-biology-math-contract|AI Computational Biology Math contract]]
- [[papers/workflows/ai-molecular-math-readiness-gate|AI Computational Biology Math readiness gate]]
- [[math/formula-explanation-ladder|Formula explanation ladder]]
- [[concepts/coverage-matrix|Coverage matrix]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
