---
title: Discrete Math and Graphs
tags:
  - math
  - graphs
  - discrete-math
---

# Discrete Math and Graphs

이산수학은 token, set, graph, tree, mask, index, neighborhood, retrieval candidate, search space를 다루는 언어입니다. 특히 [[concepts/architectures/gnn|Graph neural networks]], molecular graph, protein contact map, agent workflow를 이해할 때 중요합니다.

## Route Map

| 질문 | 시작점 | 쓰임 |
| --- | --- | --- |
| graph object가 무엇인가? | [Graph](/concepts/modalities/graph), [Graph construction](/concepts/architectures/graph-construction) | node, edge, adjacency, feature |
| model이 graph structure를 어떻게 쓰는가? | [Graph neural networks](/concepts/architectures/gnn), [Graph Transformer](/concepts/architectures/graph-transformer) | message passing, relation-aware attention |
| 입력이 unordered set이면 무엇이 달라지는가? | [Deep Sets](/concepts/architectures/deep-sets), [Set Transformer](/concepts/architectures/set-transformer) | permutation-invariant 또는 equivariant processing |
| computational biology에서는 어디에 나타나는가? | [Molecular graph](/concepts/molecular-modeling/molecular-graph), [Contact map](/concepts/protein-modeling/contact-map) | atom, residue, bond, spatial contact |

## Graph Objects

Graph는 보통 다음처럼 씁니다.

$$
G=(V,E)
$$

$V$는 node set, $E$는 edge set입니다. Node feature와 edge feature가 있으면 다음처럼 둡니다.

$$
X \in \mathbb{R}^{|V|\times d_v},
\quad
E_{ij} \in \mathbb{R}^{d_e}
$$

Adjacency matrix는 연결성을 기록합니다.

$$
A_{ij}
=
\begin{cases}
1 & (i,j)\in E \\
0 & \text{otherwise}
\end{cases}
$$

## Neighborhoods

Message passing은 보통 node neighborhood 위에서 정보를 aggregate합니다.

$$
\mathcal{N}(i)
=
\{j \mid (j,i)\in E\}
$$

일반적인 update는 다음처럼 쓸 수 있습니다.

$$
h_i^{(t+1)}
=
\phi
\left(
h_i^{(t)},
\operatorname{AGG}_{j\in\mathcal{N}(i)}
\psi(h_i^{(t)},h_j^{(t)},e_{ij})
\right)
$$

Aggregation은 보통 neighbor 순서에 대해 permutation-invariant해야 합니다.

## Sets and Permutations

Set function $f(\{x_1,\ldots,x_n\})$에서는 입력 순서가 결과를 바꾸지 않아야 합니다.

$$
f(x_1,\ldots,x_n)
=
f(x_{\pi(1)},\ldots,x_{\pi(n)})
$$

이는 임의의 permutation $\pi$에 대해 성립해야 합니다. 이 조건이 set model, pooling, readout, graph aggregator의 핵심 제약입니다.

## Paths and Connectivity

Shortest path, connected component, graph distance, neighborhood radius는 어떤 정보가 특정 node까지 도달할 수 있는지 정의합니다.

$$
\operatorname{dist}_G(i,j)
=
\text{length of the shortest path from } i \text{ to } j
$$

$L$ layer message passing GNN에서 node $i$는 보통 최대 $L$-hop neighborhood의 정보까지만 받을 수 있습니다.

## Computational Biology Connections

- Molecular structure: atom을 node로, bond 또는 spatial contact를 edge로 둡니다.
- Protein modeling: residue를 node로, contact map 또는 distance cutoff를 edge로 둡니다.
- Structure-based modeling: ligand, pocket, interaction graph를 다룹니다.
- Retrieval: candidate, neighborhood, ranking set, graph-based index를 다룹니다.

## Checks

- node 단위가 atom, residue, token, document, tool, state 중 무엇인가?
- edge가 chemical bond, spatial contact, sequence adjacency, learned attention, retrieval link 중 무엇인가?
- graph가 directed, undirected, weighted, dynamic, heterogeneous 중 어디에 해당하는가?
- 어떤 연산이 permutation-invariant 또는 permutation-equivariant해야 하는가?
- model에 local neighborhood, global attention, 또는 둘 다 필요한가?

## Related

- [[math/index|Math]]
- [[math/geometry-symmetry|Geometry and symmetry]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/protein-modeling/contact-map|Contact map]]
