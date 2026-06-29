---
title: Linear Algebra
tags:
  - math
  - linear-algebra
---

# Linear Algebra

Linear algebra는 vector, matrix, embedding, projection, learned representation을 다루는 언어입니다.

$$
y = Wx + b
$$

이 식은 [[concepts/architectures/linear-layer|linear layer]], 많은 classifier, 더 큰 architecture 안의 local transformation 뒤에 있는 기본 형태입니다.

## Route Map

| Question | Start | Use for |
| --- | --- | --- |
| 기본 vector/matrix object는 무엇인가? | [Linear algebra](/concepts/math/linear-algebra), [Linear layer](/concepts/architectures/linear-layer) | affine transform, projection, classifier, feature mixing |
| similarity를 어떻게 측정하는가? | [Vector norm and similarity](/concepts/math/vector-norm-similarity), [Embedding](/concepts/architectures/embedding) | retrieval, clustering, representation diagnostics |
| 어떤 방향이 variance나 dynamics를 설명하는가? | [Eigenvalue and eigenvector](/concepts/math/eigenvalue-eigenvector), [SVD](/concepts/math/singular-value-decomposition) | PCA, low-rank structure, spectral view |
| feature들이 함께 어떻게 변하는가? | [Covariance and correlation](/concepts/math/covariance-correlation) | normalization, redundancy, representation analysis |

## Shape Discipline

AI에서 쓰는 linear algebra는 대부분 tensor linear algebra입니다. 같은 operation도 어떤 axis를 섞는지에 따라 의미가 달라집니다.

흔한 shape는 아래와 같습니다.

$$
X_{\mathrm{batch}}
\in
\mathbb{R}^{B\times d}
$$

$$
X_{\mathrm{sequence}}
\in
\mathbb{R}^{B\times T\times d}
$$

$$
X_{\mathrm{graph}}
\in
\mathbb{R}^{B\times N\times d}
$$

$$
X_{\mathrm{coords}}
\in
\mathbb{R}^{N\times 3}
$$

여기서 $B$는 batch size, $T$는 sequence length, $N$은 node, atom, residue 수, $d$는 feature dimension입니다.

핵심 질문은 어떤 axis가 섞이는가입니다.

- Feature mixing: linear layer와 MLP는 $d$ dimension을 섞습니다.
- Token mixing: attention은 attention matrix를 통해 position을 섞습니다.
- Node mixing: graph model은 edge나 adjacency를 통해 neighborhood를 섞습니다.
- Coordinate transform: rotation은 3D coordinate axis에 작용합니다.

## Common Operations

| Operation | Formula | AI use |
| --- | --- | --- |
| Matrix-vector product | $y=Wx$ | linear layer, classifier, projection |
| Matrix-matrix product | $Y=XW$ | batched feature mixing |
| Dot product | $x^\top y$ | similarity, attention score |
| Norm | $\lVert x\rVert_2$ | distance, normalization, regularization |
| Outer product | $xy^\top$ | covariance, attention-style pair matrix |
| Trace | $\operatorname{tr}(A)$ | matrix identities, covariance, loss simplification |

## Projection

Projection은 object를 유용한 subspace로 보냅니다. Linear projection은 아래와 같습니다.

$$
z = xW
$$

여기서 $W$는 representation basis 또는 dimension을 바꿉니다. AI에서 projection은 아래 형태로 자주 등장합니다.

- embedding projection
- query, key, and value projection in [[concepts/architectures/attention|Attention]]
- PCA projection for analysis
- low-dimensional bottlenecks
- task heads and linear probes

Basis matrix $U$가 있는 subspace로의 orthogonal projection은 아래처럼 쓸 수 있습니다.

$$
P_U x
=
UU^\top x
$$

$U$의 column이 orthonormal일 때 성립합니다.

## Rank and Low-Rank Structure

Matrix rank는 해당 matrix가 표현할 수 있는 independent direction 수를 측정합니다.

$$
\operatorname{rank}(W)
\le
\min(d_{\mathrm{in}},d_{\mathrm{out}})
$$

Low-rank structure는 아래에서 등장합니다.

- PCA and SVD diagnostics
- embedding compression
- low-rank adapters such as LoRA-style updates
- representation collapse checks
- protein or molecule similarity matrices with redundant structure

Low-rank factorization은 큰 matrix를 두 개의 작은 matrix로 대체합니다.

$$
W
\approx
AB,
\quad
A\in\mathbb{R}^{d_{\mathrm{in}}\times r},
\quad
B\in\mathbb{R}^{r\times d_{\mathrm{out}}},
\quad
r \ll \min(d_{\mathrm{in}},d_{\mathrm{out}})
$$

이 방식은 parameter를 줄이지만, 표현 가능한 transformation도 제한합니다.

## Attention as Linear Algebra

Self-attention은 대부분 matrix multiplication과 softmax로 볼 수 있습니다.

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
S
=
\frac{QK^\top}{\sqrt{d_k}}
$$

$$
Y
=
\operatorname{softmax}(S)V
$$

여기서 $S$는 element 사이의 similarity matrix입니다. 단순한 수식 세부 사항이 아니라 어떤 token, residue, atom, retrieved chunk가 정보를 교환할 수 있는지를 정의합니다.

## Structure-Based Coordinate Matrices

Structure-based modeling은 coordinate를 보통 아래처럼 표현합니다.

$$
X
=
\begin{bmatrix}
x_1^\top \\
x_2^\top \\
\vdots \\
x_N^\top
\end{bmatrix}
\in
\mathbb{R}^{N\times 3}
$$

A rigid transform is:

$$
X'
=
XR^\top + \mathbf{1}t^\top
$$

여기서 $R\in\mathbb{R}^{3\times 3}$는 rotation matrix, $t\in\mathbb{R}^{3}$는 translation vector입니다.

이 관점은 linear algebra를 [[math/geometry-symmetry|Geometry and symmetry]], [[concepts/geometric-deep-learning/equivariance|Equivariance]], structure evaluation과 연결합니다.

## AI Connections

- Embedding similarity는 dot product, cosine similarity, vector norm을 사용합니다.
- Attention score는 softmax 전에 matrix product를 사용합니다.
- PCA, low-rank structure, representation diagnostic은 eigenvector와 SVD를 사용합니다.
- Linear probe는 representation이 task-relevant information을 이미 담고 있는지 test합니다.
- Coordinate model은 position, rotation, rigid transform, distance geometry에 matrix를 사용합니다.
- GPU bottleneck은 큰 matrix multiplication과 memory layout에서 자주 생깁니다.

## Checks

- vector와 matrix의 shape가 무엇인가?
- transformation이 feature, token, node, channel 중 무엇을 섞는가?
- projection이 dimension reduction, basis change, Q/K/V feature 생성 중 무엇을 하는가?
- similarity가 normalized되어 있는가, scale-sensitive한가?
- low-rank approximation이 model assumption인가, analysis tool인가?
- coordinate transform이 distance와 angle을 보존하는가?
- matrix multiplication이 compute-bound, memory-bound, communication-bound 중 무엇인가?

## Related

- [[math/index|Math]]
- [[ai/architectures|Architectures]]
- [[math/geometry-symmetry|Geometry and symmetry]]
- [[math/numerical-computing|Numerical computing]]
- [[concepts/learning/linear-probing|Linear probing]]
