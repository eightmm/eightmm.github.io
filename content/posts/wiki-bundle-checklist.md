---
title: Wiki Bundle Checklist
tags:
  - posts
  - writing
  - workflow
---

# Wiki Bundle Checklist

Korean post는 독자가 들어오는 입구이고, wiki note는 반복해서 참조할 지식 단위입니다. 그래서 post를 먼저 길게 쓰기보다, 아래 bundle이 어느 정도 갖춰졌는지 확인한 뒤 글로 승격합니다.

$$
\text{post bundle}
=
\text{question}
+ \text{object}
+ \text{method}
+ \text{formula}
+ \text{evidence}
+ \text{next path}
$$

## Minimum Bundle

| Part | Required note | Pass when |
| --- | --- | --- |
| Reader question | [[posts/blog-writing-guide|Blog writing guide]] | 글이 한 문장 질문에 답한다 |
| Route | [[concepts/coverage-matrix|Coverage matrix]] | 중심축이 AI, Computational Biology, Math, Papers, Projects, Infra, Agents 중 하나로 정해진다 |
| Object | [[entities/index|Entities]] or [[molecular-modeling/entities|Computational biology entities]] | 무엇을 모델링하는지 명확하다 |
| Representation | [[concepts/modalities/representation-contract|Representation contract]] | raw object가 token, graph, coordinate, embedding, sample로 바뀌는 과정이 보인다 |
| Method | [[ai/index|AI]] | architecture, learning method, generative model, evaluation 중 무엇이 핵심인지 분리된다 |
| Formula | [[math/formula-intake|Formula intake]] | 필요한 수식과 모든 symbol이 설명된다 |
| Evidence | [[concepts/evaluation/claim-evidence-boundary|Claim-evidence boundary]] | benchmark, split, metric, baseline, leakage, uncertainty의 경계가 있다 |
| Paper source | [[papers/workflows/paper-review-workflow|Paper review workflow]] | 특정 논문 claim은 paper note나 evidence table로 분리된다 |
| Extraction | [[papers/workflows/paper-to-wiki-extraction|Paper to wiki extraction]] | paper claim이 object, method, formula, evidence update로 나뉜다 |
| Public boundary | [[inbox/publishing-gate|Publishing gate]] | private server, account, port, path, collaborator, unpublished result가 없다 |
| Next path | [[posts/post-promotion-gate|Post promotion gate]] | 독자가 다음에 볼 3-7개 note가 있다 |

## Post Type Bundles

| Post type | Bundle에 포함할 것 |
| --- | --- |
| AI architecture map | architecture note, input modality, complexity, inductive bias, evaluation risk |
| Learning method explainer | objective, data signal, representation, transfer setting, failure mode |
| Generative model explainer | modeled distribution, sampling path, objective, evaluation metric, validity checks |
| Computational biology object post | entity note, representation contract, label context, split unit, leakage risk |
| Docking or SBDD post | protein-ligand object, pose/scoring concept, preparation boundary, benchmark/evaluation note |
| Protein or sequence post | sequence/structure object, representation, family split, task output, evidence boundary |
| Math-heavy post | formula pattern, symbol table, operation meaning, objective-metric link, example use |
| Paper cluster post | individual paper notes, claim-evidence table, benchmark card, updated concept notes |
| Project post | project note, problem definition, design decision, verification, public artifact boundary |
| Infra post | public runbook, failure taxonomy, reproducibility note, no operational secrets |
| Agent workflow post | workflow note, tool/memory boundary, verification loop, handoff protocol |

## Formula Rule

Equation이 topic을 더 이해하기 쉽게 만든다면 포함합니다. Post에서는 equation을 짧게 둘 수 있지만 core operation을 숨기면 안 됩니다.

예를 들어 attention을 `Attn(x)`로만 줄이면 부족합니다. Post 수준에서 유용한 형태는 아래와 같습니다.

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

그 다음 $X$, $Q$, $K$, $V$, $d_k$, softmax가 적용되는 axis를 정의합니다. 더 긴 derivation은 [[math/formula-intake|Formula intake]] 또는 관련 Math note에 둡니다.

## When To Create Wiki Notes First

아래 경우에는 post를 쓰기 전에 wiki note를 먼저 만들거나 업데이트합니다.

- post가 같은 term을 한 paragraph 이상 정의해야 할 때.
- claim이 symbol table이 없는 formula에 의존할 때.
- result가 dataset split, benchmark, leakage risk에 의존할 때.
- topic이 AI, Computational Biology, Math를 가로지르지만 route page가 없을 때.
- post의 핵심 주장을 만들기 위해 paper가 여러 개 필요할 때.
- reader가 다음에 무엇을 읽어야 할지 알기 어려울 때.

## Naming Rule

Sequence, structure, molecule, ligand, docking, conformer, genome-level object가 함께 나올 수 있으면 public umbrella로 `Computational Biology`를 씁니다. `Molecular Modeling`은 molecule, conformer, docking, structure-heavy subset에만 씁니다. Model family와 learning method는 `AI`, formula와 abstraction은 `Math`에 둡니다.

## Related

- [[posts/ai-molecular-math-post-intake|AI Computational Biology Math post intake]]
- [[posts/post-promotion-gate|Post promotion gate]]
- [[posts/synthesis-post-template|Synthesis post template]]
- [[papers/workflows/paper-to-wiki-extraction|Paper to wiki extraction]]
- [[concepts/coverage-matrix|Coverage matrix]]
- [[concepts/ai-computational-biology-math-contract|AI Computational Biology Math contract]]
- [[ai/paper-claim-patterns|AI paper claim patterns]]
- [[molecular-modeling/paper-claim-patterns|Computational Biology paper claim patterns]]
- [[math/formula-patterns|Formula pattern catalog]]
