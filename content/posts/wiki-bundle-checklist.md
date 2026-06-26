---
title: Wiki Bundle Checklist
tags:
  - posts
  - writing
  - workflow
---

# Wiki Bundle Checklist

Korean post는 독자가 들어오는 입구이고, English wiki note는 반복해서 참조할 지식 단위입니다. 그래서 post를 먼저 길게 쓰기보다, 아래 bundle이 어느 정도 갖춰졌는지 확인한 뒤 글로 승격합니다.

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

| Part | Required Note | Pass When |
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

| Post Type | Bundle Should Include |
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

If an equation makes the topic easier to understand, include it. A post can keep the equation short, but it should not hide the core operation.

For example, attention should not be reduced to `Attn(x)`. The useful post-level form is:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Then define $X$, $Q$, $K$, $V$, $d_k$, and the axis over which softmax is applied. Longer derivations belong in [[math/formula-intake|Formula intake]] or the relevant Math note.

## When To Create Wiki Notes First

Create or update wiki notes before writing a post when:

- the post would define the same term for more than one paragraph;
- the claim depends on a formula that has no symbol table;
- the result depends on a dataset split, benchmark, or leakage risk;
- the topic crosses AI, Computational Biology, and Math but has no route page;
- the post needs more than one paper to make the point;
- the reader would not know what to read next.

## Naming Rule

Use `Computational Biology` as the public umbrella when sequence, structure, molecule, ligand, docking, conformer, and genome-level objects can appear together. Use `Molecular Modeling` only for the molecule, conformer, docking, and structure-heavy subset. Use `AI` for model families and learning methods, and `Math` for formulas and abstractions.

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
