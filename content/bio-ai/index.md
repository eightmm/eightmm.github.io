---
title: Bio-AI
tags:
  - bio-ai
---

# Bio-AI

Bio-AI 영역을 구조 기반 모델링, 단백질, 분자, 리간드, protein-ligand interaction 중심으로 정리하는 입구입니다. 전체 computational biology를 다루기보다, 실제로 다룰 가능성이 높은 구조 기반 AI와 단백질/분자 모델링에 범위를 좁힙니다.

이 페이지는 한글 안내 페이지입니다. 링크된 `concepts/*`, `entities/*`, `research/*` 문서는 영어 canonical wiki note로 유지합니다.

반복해서 등장하는 기본 형태는 생물학적/화학적 객체와 context를 모델 입력으로 두는 것입니다.

$$
\hat{y} = f_\theta(x_{\mathrm{bio}}, x_{\mathrm{context}})
$$

여기서 $x_{\mathrm{bio}}$는 sequence, molecule, structure, complex일 수 있고, $x_{\mathrm{context}}$는 pocket, target condition, assay context일 수 있습니다.

## 다루는 객체

- [[entities/protein|Protein]]
- [[entities/ligand|Ligand]]
- [[entities/molecule|Molecule]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[entities/sequence|Sequence]]
- [[entities/structure|Structure]]

## Structure-Based AI

- [[research/structure-based-ai/index|Structure-based AI]]
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[papers/sbdd/posebusters|PoseBusters]]

## Protein and Sequence Modeling

- [[research/protein-modeling/index|Protein modeling]]
- [[research/protein-modeling/mambafold|MambaFold]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]

## Geometry and Evaluation

- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/evaluation/leakage|Leakage]]

## 관련 입구

- [[ai/index|AI]]
- [[papers/index|Papers]]
- [[projects/index|Projects]]
