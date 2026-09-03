---
title: Equivariant Sheaf Neural Networks — Learning Geometric Transport on Graphs
aliases:
  - papers/esnn
  - papers/architectures/esnn
  - papers/equivariant-sheaf-neural-networks
tags:
  - papers
  - architectures
  - geometric-deep-learning
  - graph-neural-networks
  - equivariance
  - sheaf-neural-networks
  - scientific-ml
status: full-note
source_type: ArXiv
source_url: https://arxiv.org/abs/2608.28853
---

# Equivariant Sheaf Neural Networks: Learning Geometric Transport on Graphs

> **한 줄 요약:** ESNN의 가장 중요한 아이디어는 equivariant model의 표현력을 더 높은-order representation으로만 키우지 않고, **edge가 vector feature를 어떻게 운반하는가**를 directed matrix-valued transport로 학습해 relation/operator 쪽에 capacity를 배치하는 것입니다.

## 왜 이 논문을 저장하는가

3D geometric model을 강하게 만드는 가장 익숙한 방법은 hidden representation을 더 풍부하게 만드는 것입니다. 예를 들어 scalar만 쓰던 모델에 vector를 넣고, vector에서 다시 higher-order tensor 또는 더 높은 spherical-harmonic degree를 추가할 수 있습니다.

ESNN은 이 축과 직교하는 질문을 던집니다.

> **node가 무엇을 들고 있는가를 계속 복잡하게 만들기 전에, 한 node의 geometric feature가 다른 node로 이동할 때 적용되는 transport operator를 더 expressive하게 만들 수 있는가?**

이 관점은 특히 first-order equivariant model을 설계할 때 중요합니다. 단순한 scalar edge gate는 vector 전체를 같은 비율로 키우거나 줄일 뿐, edge 방향과 그 수직 방향을 서로 다르게 취급하지 못합니다. 반면 matrix-valued transport는 vector의 radial/tangential component에 서로 다른 작용을 줄 수 있고, learned covariant context가 있으면 더 풍부한 방향 의존 변환으로 확장할 수 있습니다.

따라서 이 논문의 장기적 가치는 특정 benchmark 수치보다 다음 설계 대비에 있습니다.

$$
\text{representation-centric scaling}
\quad\text{vs.}\quad
\text{operator-centric scaling}.
$$

## Metadata

| Field | Value |
| --- | --- |
| Paper | Equivariant Sheaf Neural Networks: Learning Geometric Transport on Graphs |
| Authors | Alessio Borgi, Mario Severino, Fabrizio Silvestri, Pietro Liò |
| Version | arXiv:2608.28853v1 |
| Submitted | 2026-08-28 |
| Subjects | cs.LG, cs.AI |
| Official code | **공식 ESNN implementation을 2026-09-03 기준 확인하지 못함** |
| Core claim | first-order scalar/vector carrier를 유지하면서 directed matrix-valued edge transport를 학습 |
| Symmetry | exact Euclidean equivariance; $O(n)$ transport characterization + optional controlled symmetry relaxation |
| Evaluation families | particle dynamics, mesh simulation, point-cloud classification, molecular property prediction |

이 노트는 공개된 v1의 확인 가능한 주장만 사용합니다. 정확한 benchmark 숫자나 training hyperparameter는 독립적으로 검증하지 못한 항목을 억지로 채우지 않습니다.

## 1. 문제: first-order equivariant message passing의 병목은 어디에 있는가

좌표가 있는 graph를 생각합시다.

$$
G=(V,E), \qquad x_i\in\mathbb{R}^n.
$$

global Euclidean transform을 적용하면 좌표는

$$
x_i' = Qx_i+t,
$$

처럼 변합니다. 여기서 $Q\in O(n)$이고 $t$는 translation입니다.

scalar feature $s_i$는 좌표계가 바뀌어도 그대로여야 합니다.

$$
s_i' = s_i.
$$

반면 vector feature $v_i\in\mathbb{R}^n$는

$$
v_i' = Qv_i
$$

처럼 변해야 합니다.

이제 node $j$의 vector를 node $i$로 전달하는 가장 단순한 형태를 생각하면

$$
m_{ij}^{(v)}
=
\alpha_{ij}v_j
$$

처럼 invariant scalar $\alpha_{ij}$로 vector 전체를 gate할 수 있습니다. 이 방식은 equivariance를 유지하기 쉽지만 transport 자체의 자유도는 작습니다.

vector $v_j$가 edge 방향에 평행한 성분과 수직인 성분을 동시에 가지고 있어도 $\alpha_{ij}$ 하나는 둘을 같은 방식으로 처리합니다.

ESNN의 출발점은 이것입니다.

$$
m_{ij}^{(v)}
=
T_{ij}v_j,
$$

여기서 $T_{ij}\in\mathbb{R}^{n\times n}$는 edge-conditioned transport operator입니다.

핵심 문제는 아무 matrix나 쓰면 equivariance가 즉시 깨진다는 것입니다. 따라서 실제 질문은:

> **$T_{ij}$를 얼마나 expressive하게 만들면서도 coordinate frame 변화와 commute하게 만들 수 있는가?**

입니다.

## 2. 최소 배경: equivariant linear operator가 만족해야 하는 조건

edge의 relative displacement를

$$
r_{ij}=x_j-x_i
$$

라고 두겠습니다.

translation은 사라지고 rotation/reflection 아래에서는

$$
r_{ij}'=Qr_{ij}
$$

입니다.

transport가 vector를 vector로 보낸다면 frame이 바뀐 뒤의 결과가 먼저 transport하고 회전한 것과 같아야 합니다.

이를 operator 수준에서 쓰면

$$
T(Qr)
=
QT(r)Q^\top
$$

와 같은 conjugation-equivariance condition이 필요합니다.

그러면

$$
T(Qr)(Qv)
=
QT(r)v
$$

가 되어 vector message 자체가 올바르게 transform합니다.

이 식은 ESNN을 이해할 때 가장 중요한 contract입니다. scalar weight가 invariant하면 충분했던 경우와 달리, matrix-valued transport는 **operator 자체가 어떻게 변해야 하는지**까지 규정해야 합니다.

## 3. 왜 radial / tangential decomposition이 자연스럽게 나오는가

$r\neq 0$일 때 unit edge direction을

$$
\hat r=\frac{r}{\|r\|}
$$

라고 두고 두 projector를 정의합니다.

$$
P_{\parallel}
=
\hat r\hat r^\top,
$$

$$
P_{\perp}
=
I-\hat r\hat r^\top.
$$

$P_{\parallel}$은 vector를 edge 방향 성분으로 보내고, $P_{\perp}$는 그 수직 공간 성분만 남깁니다.

vector도

$$
v
=
v_{\parallel}+v_{\perp},
$$

$$
v_{\parallel}
=
P_{\parallel}v,
\qquad
v_{\perp}
=
P_{\perp}v
$$

로 나뉩니다.

논문의 핵심 theoretical characterization은 **relative displacement만이 covariant geometric input일 때 linear $O(n)$-equivariant map이 radial과 tangential component에 대한 독립 작용으로 분해된다**는 것입니다.

이 결과를 이해하는 가장 유용한 canonical form은

$$
T(r)
=
a(\|r\|)P_{\parallel}
+
b(\|r\|)P_{\perp}
$$

입니다.

그러면 transport는

$$
T(r)v
=
a(\|r\|)v_{\parallel}
+
b(\|r\|)v_{\perp}
$$

가 됩니다.

즉 하나의 isotropic scalar gate와 달리 최소한 다음 둘을 독립적으로 학습할 수 있습니다.

- edge를 따라 흐르는 radial information;
- edge에 수직인 tangential information.

이것이 왜 중요한지는 간단한 극단에서 보입니다.

### isotropic scalar transport

$$
a=b.
$$

이 경우

$$
T(r)=aI
$$

이고 radial/tangential 구분이 사라집니다.

### anisotropic but symmetry-preserving transport

$$
a\neq b.
$$

그러면 edge라는 **data-defined direction**을 기준으로 vector를 다르게 처리하지만 global coordinate frame에 의존하지 않습니다.

ESNN의 중요한 메시지는 `anisotropy = symmetry breaking`이 아니라는 점입니다. 방향 의존성이 데이터의 covariant quantity에서 만들어지고 transformation law를 지키면 anisotropic operator도 정확히 equivariant할 수 있습니다.

## 4. 왜 matrix-valued edge transport인가

보통 GNN에서 edge는 다음과 같은 역할 중 하나를 합니다.

- adjacency / neighborhood를 정의;
- scalar weight를 제공;
- invariant edge embedding을 message MLP에 제공.

ESNN은 edge를 **local transport rule**로 봅니다.

$$
j
\xrightarrow{\;T_{ij}\;}
i.
$$

이때 $T_{ij}$와 $T_{ji}$가 반드시 같은 operator일 이유가 없습니다. 따라서 directed transport는 관계의 orientation을 명시적으로 유지할 수 있습니다.

이 관점에서 중요한 distinction은:

$$
\text{edge feature}
\neq
\text{edge transport}.
$$

edge feature는 message를 만드는 입력일 수 있지만, transport는 **feature space에 실제로 작용하는 map**입니다.

first-order vector carrier를 유지하면서도 interaction을 풍부하게 만들 수 있다는 점에서, 이 아이디어는 representation을 키우는 것과 다른 capacity axis를 제공합니다.

## 5. Sheaf 관점은 무엇을 더해주는가

cellular sheaf의 최소 직관만 가져오면 각 node의 feature를 그대로 같은 공간에서 비교한다고 가정하지 않고, 관계를 통과할 때 local linear map을 적용합니다.

일반적인 sheaf language에서는 node/edge에 local vector space가 있고 restriction map이 이 공간 사이의 compatibility를 정의합니다.

ESNN에서 기억해야 할 핵심은 추상적 sheaf formalism 전체가 아니라:

> **neighbor feature를 '그대로 복사해서 합산'하는 대신, 관계에 맞는 transport를 거쳐 비교·집계한다.**

는 사고방식입니다.

이것은 geometric graph에서 특히 자연스럽습니다. node마다 vector feature가 있고 edge마다 relative geometry가 다르기 때문에, 모든 방향을 동일하게 처리하는 scalar coupling보다 relation-specific operator를 쓰는 것이 더 풍부한 local geometry를 표현할 수 있습니다.

다만 `sheaf`라는 이름만 보고 모든 gauge-equivariant sheaf architecture의 성질이 자동으로 따라온다고 해석하면 안 됩니다. 이 논문에서 실제로 검증해야 할 것은 **제안된 transport parameterization과 symmetry contract**입니다.

## 6. learned covariant feature가 왜 더 큰 표현력을 주는가

relative displacement $r_{ij}$만 있으면 edge가 제공하는 distinguished direction은 사실상 $\hat r_{ij}$ 하나뿐입니다.

그 경우 full $O(n)$ symmetry 아래에서 만들 수 있는 linear vector-to-vector transport가 radial/tangential decomposition으로 강하게 제한되는 것은 자연스럽습니다.

그런데 network가 추가 covariant vector $c_{ij}$를 학습한다면 상황이 달라집니다.

$$
c_{ij}'=Qc_{ij}.
$$

이제 operator는

$$
T(r_{ij},c_{ij},\ldots)
$$

처럼 여러 geometric direction에 조건화될 수 있습니다.

올바른 equivariance를 위해서는 여전히

$$
T(Qr,Qc,\ldots)
=
QT(r,c,\ldots)Q^\top
$$

를 만족해야 합니다.

하지만 basis를 만들 수 있는 covariant ingredient가 늘어나므로 $\hat r\hat r^\top$ 하나로 표현되지 않는 richer matrix structure를 구성할 수 있습니다.

여기서 논문의 architectural thesis가 더 분명해집니다.

$$
\text{higher-order node state}
\quad\text{없이도}\quad
\text{richer geometric operator}
$$

를 만들 수 있다는 것입니다.

중요한 caveat도 있습니다. 이것은 higher-order irreps가 불필요하다는 증명이 아닙니다. higher-order representation은 다른 종류의 angular information과 composition rule을 제공합니다. ESNN은 **대체 가능성**보다는 **보완적인 expressivity axis**를 제시한다고 읽는 편이 안전합니다.

## 7. controlled symmetry relaxation: 언제 equivariance를 일부러 약하게 만드는가

모든 physical system이 full Euclidean symmetry를 가지는 것은 아닙니다.

예를 들어 gravity가 있는 환경에서는 특정 ambient direction이 특별합니다. 이때 실제 data-generating process는 arbitrary rotation 아래 동일하지 않을 수 있습니다.

논문은 이런 setting을 위해 preferred direction을 사용할 수 있는 controlled symmetry relaxation을 도입하고, direction이 prescribed 또는 inferred될 수 있다고 설명합니다. directional pathway가 비활성화되면 full $E(n)$ equivariance를 회복하는 것이 design boundary입니다.

이 부분은 실무에서 매우 조심해서 써야 합니다.

### 올바른 사용

task 자체에 물리적인 preferred axis가 존재하고 그 정보가 deployment에서도 실제로 주어집니다.

### 위험한 사용

dataset의 coordinate convention을 preferred direction으로 학습해 benchmark shortcut을 먹습니다.

따라서 symmetry relaxation을 평가할 때는 단순 ID score보다 다음이 중요합니다.

- rotation stress test;
- preferred direction을 제거했을 때의 성능;
- direction을 shuffle했을 때의 성능;
- train frame과 test frame을 다르게 했을 때의 robustness.

`symmetry를 약하게 만들었다`는 사실 자체는 장점이나 단점이 아닙니다. **task symmetry와 model symmetry가 일치하는가**가 핵심입니다.

## 8. Architecture contract

| Object | Representation | Transformation |
| --- | --- | --- |
| coordinate $x_i$ | point in $\mathbb R^n$ | $x_i\mapsto Qx_i+t$ |
| relative displacement $r_{ij}$ | polar vector | $r_{ij}\mapsto Qr_{ij}$ |
| scalar node/edge feature | invariant | unchanged |
| first-order vector feature $v_i$ | vector | $v_i\mapsto Qv_i$ |
| radial projector $P_\parallel$ | rank-2 operator | $P_\parallel\mapsto QP_\parallel Q^\top$ |
| tangential projector $P_\perp$ | rank-2 operator | $P_\perp\mapsto QP_\perp Q^\top$ |
| transport $T_{ij}$ | linear operator | $T_{ij}\mapsto QT_{ij}Q^\top$ |
| transported vector $T_{ij}v_j$ | vector | $\mapsto Q(T_{ij}v_j)$ |
| optional preferred direction | covariant/external context | must match the task's symmetry contract |

이 표에서 가장 중요한 것은 $T_{ij}$가 invariant matrix가 아니라는 점입니다. coordinate matrix entry를 그대로 고정하면 안 되고 global transform과 함께 conjugate되어야 합니다.

## 9. EGNN과 무엇이 다른가

[[papers/architectures/egnn|E(n) Equivariant GNN]]은 간결한 first-order geometric model의 대표적인 anchor입니다.

EGNN류의 핵심 intuition은 invariant distance에서 scalar message를 만들고 relative coordinate vector를 scalar로 scale해 coordinate update를 만드는 것입니다.

schematic하게:

$$
\Delta x_i
\propto
\sum_j
(x_i-x_j)\,\phi(m_{ij}).
$$

이 방식은 단순하고 강력하지만 vector transformation의 주요 primitive가 `relative vector × scalar`에 가깝습니다.

ESNN의 관심은 다른 곳에 있습니다.

$$
v_j
\xrightarrow{T_{ij}}
\widetilde v_{j\to i}.
$$

즉 **이미 존재하는 vector feature를 edge-specific linear operator로 transport**합니다.

따라서 둘의 대비를 다음처럼 읽는 것이 좋습니다.

| Axis | EGNN-style intuition | ESNN-style intuition |
| --- | --- | --- |
| geometric primitive | invariant scalar + relative vector | equivariant matrix-valued transport |
| carrier | scalar hidden + coordinates 중심 | scalar + first-order vector |
| anisotropy | relative direction을 scalar-gated update에 사용 | operator가 radial/tangential component를 다르게 변환 |
| capacity location | message MLP / coordinate update | edge transport operator |
| higher-order irreps | 필요 없음 | 필요 없음이 핵심 설계 목표 |

이 비교는 우열이 아니라 **어디에 표현력을 넣는가**의 차이입니다.

## 10. higher-order irreps와의 관계

$SO(3)$/$O(3)$ irrep model은 $l=0,1,2,\ldots$ representation을 통해 angular structure를 hidden state에 직접 보존할 수 있습니다.

ESNN은 이를 계속 올리는 대신 first-order scalar/vector state에 머무르며 edge operator를 강화합니다.

그래서 가장 중요한 empirical comparison은 단순히 `ESNN vs 특정 baseline`이 아닙니다.

다음과 같은 **compute-matched expressivity comparison**이 필요합니다.

1. scalar/vector + isotropic edge gate;
2. scalar/vector + radial/tangential transport;
3. scalar/vector + learned feature-conditioned matrix transport;
4. $l=2$까지 확장한 higher-order representation.

그리고 최소한 다음 budget을 함께 맞춰야 합니다.

- parameters;
- FLOPs 또는 executed operations;
- wall-clock ms/step;
- peak memory;
- neighborhood size.

그렇지 않으면 `operator가 더 좋다`와 `그냥 더 많은 compute를 썼다`를 분리할 수 없습니다.

## 11. complexity를 어떻게 봐야 하는가

matrix-valued transport는 무료가 아닙니다.

단순 scalar gate는 vector channel당 작은 수의 coefficient만 필요하지만, 일반 dense map은 channel structure에 따라 훨씬 많은 coefficient와 multiply를 요구할 수 있습니다.

논문의 theoretical decomposition은 여기서 실용적인 의미를 가집니다.

$$
T(r)
=
aP_\parallel+bP_\perp
$$

처럼 structure를 강제하면 arbitrary dense matrix를 직접 예측하지 않고도 anisotropic transport를 얻을 수 있습니다.

따라서 실제 구현에서 질문해야 할 것은:

- full matrix를 직접 parameterize하는가;
- radial/tangential basis coefficient만 예측하는가;
- learned covariant feature로 basis를 얼마나 늘리는가;
- channel mixing과 spatial $n\times n$ transport를 어떻게 분리하는가.

현재 공개 정보만으로 exact wall-clock advantage를 일반화할 수는 없습니다. `higher-order representation보다 효율적`이라는 방향성은 plausible하지만, **동일 hardware의 matched benchmark가 없으면 cost 우위는 독립적인 claim**으로 남겨야 합니다.

## 12. evaluation contract: 네 가지 task family가 각각 무엇을 검증하는가

논문은 하나의 benchmark에만 의존하지 않고 서로 다른 geometric failure mode를 가진 task family를 사용했다고 보고합니다.

| Evaluation family | 검증하는 핵심 질문 | 이 결과만으로 말할 수 없는 것 |
| --- | --- | --- |
| Particle dynamics | vector transport가 dynamics와 rollout에 도움이 되는가 | biomolecular interaction generalization |
| Mesh simulation | local directional transport와 long-horizon stability가 유용한가 | arbitrary graph에서의 universal gain |
| Point-cloud classification | frame 변화, 특히 unseen rotation에 robust한가 | physical dynamics fidelity |
| Molecular property prediction | first-order geometric representation이 invariant molecular target에 유용한가 | protein–ligand pose/affinity/screening utility |

저자들은 dynamics prediction 개선, symmetry가 실제로 깨진 setting에서 gravity axis recovery, 일부 mesh task 및 long-horizon rollout 개선, unseen rotations에 대한 robustness를 보고합니다.

여기서 숫자를 과도하게 요약하지 않는 이유는 간단합니다. 이 노트의 핵심 claim은 `새 benchmark SOTA`가 아니라 **transport operator라는 architecture axis가 실제 여러 geometric domain에서 작동하는가**이기 때문입니다.

## 13. evidence를 어떻게 읽어야 하는가

현재 evidence가 지지하는 가장 강한 문장은 다음 정도입니다.

> **first-order scalar/vector representation 안에서도 edge transport를 matrix-valued equivariant operator로 풍부하게 만드는 것이 여러 geometric task에서 유용할 수 있다.**

반면 다음은 아직 지지되지 않습니다.

### “higher-order irreps는 필요 없다”

아닙니다. ESNN은 higher-order representation을 쓰지 않는 complementary route를 보여줄 뿐, 모든 angular task에서 동등하거나 우월하다고 증명하지 않습니다.

### “protein–ligand modeling에도 바로 더 좋다”

직접 evidence가 없습니다. molecular property prediction과 protein–ligand interaction은 task structure가 다릅니다.

### “matrix transport가 더 빠르다”

현재 확인된 public evidence만으로 universal wall-clock 우위를 말할 수 없습니다.

### “symmetry relaxation이 항상 도움이 된다”

preferred direction이 실제 task symmetry에 존재할 때만 정당화됩니다. coordinate convention shortcut이면 오히려 잘못된 inductive bias입니다.

## 14. 가장 중요한 ablation: 표현력의 source를 분해하기

이 논문을 재현한다면 전체 ESNN과 baseline 하나만 비교하지 않습니다.

가장 먼저 다음 ladder를 만듭니다.

### A. isotropic scalar gate

$$
T_{ij}
=
a_{ij}I.
$$

### B. radial / tangential transport

$$
T_{ij}
=
a_{ij}P_{\parallel,ij}
+
b_{ij}P_{\perp,ij}.
$$

### C. feature-conditioned transport

learned covariant context를 추가해 더 풍부한 equivariant operator basis를 만듭니다.

### D. higher-order carrier baseline

동일 compute budget에서 $l=2$ 등의 representation을 추가합니다.

이 ablation의 목적은:

$$
\text{gain}
=
\text{anisotropy}
?
+
\text{directedness}
?
+
\text{learned covariant basis}
?
+
\text{extra compute}
?
$$

를 분리하는 것입니다.

특히 B가 A를 크게 이기고 C가 B보다 조금만 좋아진다면 `full matrix transport`보다 **radial/tangential inductive bias**가 핵심일 수 있습니다.

반대로 C가 B를 크게 이기면 learned covariant context가 실제로 추가 geometry를 담는다는 evidence가 됩니다.

## 15. symmetry correctness는 unit test가 아니라 scientific evidence다

equivariant architecture에서 symmetry test는 단순 software QA가 아닙니다. model claim의 일부입니다.

최소 numerical test는 다음과 같습니다.

### O(n) test

random orthogonal $Q$에 대해

$$
f(QX, QV)
\approx
Qf(X,V).
$$

### translation test

$$
f(X+t,V)
$$

의 invariant output은 변하지 않아야 하고 coordinate-like output은 적절히 translate되어야 합니다.

### permutation test

node permutation $\pi$에 대해

$$
f(\pi X,\pi H)
=
\pi f(X,H).
$$

### reflection test

$O(3)$을 주장한다면 $\det Q=-1$인 transform도 따로 검사해야 합니다.

### preferred-direction test

directional pathway를 끄면 full symmetry가 회복되는지 확인합니다.

이 검증을 통과하지 않은 성능 gain은 architecture의 의도와 무관한 coordinate leakage에서 왔을 가능성을 배제할 수 없습니다.

## 16. TriELA와 연결해서 읽기

공개된 [TriELA repository](https://github.com/eightmm/equivariant-linear-attention) 설계는 세 persistent stream을 둡니다.

- `H`: parity-aware node irreps;
- `Z`: ordered O(3)-invariant pair state;
- `X`: coordinates.

또한 directed pair state, exact triangle multiplication, pair-to-node update, Global ELA, pair-conditioned Local ELA를 사용합니다.

ESNN과 직접 겹치는 질문은 **pair/relation information을 node equivariant transport에 어떻게 쓰는가**입니다.

TriELA는 이미 invariant directed pair context와 equivariant node carrier를 분리합니다. ESNN 관점을 가져오면 다음 질문을 독립적으로 검증할 수 있습니다.

> pair context가 단순 scalar gate/attention coefficient를 만드는 수준을 넘어 **equivariant transport operator의 coefficient**를 만들게 하면 어떤 정보가 추가되는가?

이때 가장 깨끗한 실험은 architecture 전체를 바꾸지 않고 local relation operator만 바꾸는 것입니다.

| Variant | Node carrier | Edge / pair operator |
| --- | --- | --- |
| A | fixed low-order irreps | isotropic scalar gating |
| B | same | radial/tangential transport |
| C | same | pair-conditioned matrix transport |
| D | higher-order carrier | simpler transport |

평가할 것은:

- task metric;
- target/scaffold 또는 구조적 OOD;
- symmetry numerical error;
- convergence step;
- ms/step;
- peak GPU memory;
- params/FLOPs.

중요한 점은 현재 공개된 TriELA 설명만으로 이 실험의 결과를 미리 가정하지 않는 것입니다. ESNN은 **검증할 architecture hypothesis**를 제공하지, 이미 우월함이 입증된 replacement를 제공하는 것이 아닙니다.

## 17. protein–ligand로 옮길 때 추가로 생기는 문제

protein–ligand graph는 일반 point cloud보다 relation semantics가 복잡합니다.

최소한:

- protein local geometry;
- ligand covalent graph;
- protein–ligand cross-interface relation;
- entity/chain identity;
- ligand atom permutation;
- global $O(3)$ / translation symmetry

가 동시에 있습니다.

따라서 하나의 $T_{ij}$ family를 모든 edge에 공유하는 것이 최적이라는 보장은 없습니다.

오히려 자연스러운 확장은:

$$
T_{ij}
=
T(
r_{ij},
z_{ij},
\text{relation type}
)
$$

처럼 invariant pair context $z_{ij}$와 relation sector를 transport coefficient에 조건화하는 것입니다.

하지만 여기서 반드시 지켜야 하는 규칙이 있습니다.

$z_{ij}$가 invariant이면 transport basis의 spatial transformation law를 바꾸지 않고 coefficient만 조절할 수 있습니다. 반대로 raw Cartesian component를 invariant MLP에 그대로 넣으면 frame leakage가 생길 수 있습니다.

또한 SBDD에서 final target이 affinity 같은 invariant scalar라고 해도 hidden layer의 equivariance correctness를 생략할 수 없습니다. 잘못된 coordinate shortcut이 random split에서만 좋은 score를 낼 수 있기 때문입니다.

## 18. OOD에서 무엇을 검증해야 하는가

3D architecture의 generalization claim은 random split 하나로는 부족합니다.

protein–ligand adaptation이라면 최소한 다음 axis를 분리해야 합니다.

### protein OOD

sequence/structure family 단위 split.

### ligand OOD

Bemis–Murcko scaffold 또는 ligand similarity group split.

### joint OOD

새 protein family + 새 ligand scaffold.

### geometric stress

input frame rotation/reflection, conformer perturbation, pose perturbation.

ESNN-style operator가 정말 local geometric inductive bias를 개선한다면 ID score뿐 아니라 **frame stress와 structural OOD에서 gain이 남는지**가 중요한 evidence가 됩니다.

## 19. reproducibility path

2026-09-03 기준 이 논문의 공식 ESNN implementation은 확인하지 못했습니다. 따라서 reproducibility를 보수적으로 낮게 보는 것이 맞습니다.

현재 가능한 재현 경로는 두 단계로 나뉩니다.

### Tier 1 — theorem-inspired minimal implementation

논문 전체를 재현한다고 주장하지 않고 다음 primitive만 구현합니다.

$$
T(r)
=
a(r)P_\parallel
+
b(r)P_\perp.
$$

이 layer에 대해 O(3), reflection, translation, permutation numerical tests를 먼저 통과시킵니다.

### Tier 2 — paper reproduction

공식 code/config가 공개되면 exact architecture, dataset preprocessing, split, training budget, checkpoint selection, evaluation protocol을 맞춥니다.

공식 implementation이 없는 상태에서 임의로 만든 radial/tangential layer를 `ESNN reproduced`라고 부르면 안 됩니다. 그것은 **ESNN의 공개 theoretical claim에서 영감을 받은 independent implementation**입니다.

## 20. failure modes

| Failure mode | 왜 위험한가 |
| --- | --- |
| arbitrary dense $T_{ij}$ 예측 | conjugation law가 깨져 equivariance 상실 |
| coordinate component를 scalar MLP에 직접 입력 | frame shortcut / leakage |
| preferred direction이 dataset frame을 외움 | symmetry relaxation이 shortcut으로 변질 |
| $T_{ij}=T_{ji}$를 무조건 강제 | directed relation information을 잃을 수 있음 |
| matrix capacity만 늘리고 baseline compute를 안 맞춤 | architecture gain과 extra compute가 섞임 |
| invariant final metric만 보고 hidden symmetry test 생략 | frame-dependent internal bug를 놓침 |
| molecular property 결과를 PL interaction evidence로 확대 | task boundary 위반 |
| random split만 사용 | target/scaffold OOD claim 불가 |
| long-horizon gain만 보고 one-step error를 무시 | rollout 안정성과 local accuracy를 혼동 |
| higher-order baseline을 약하게 설정 | operator-vs-representation 결론이 불공정 |

## 21. 무엇이 이 논문을 falsify하거나 약하게 만드는가

좋은 architecture idea는 성공 조건뿐 아니라 실패 조건도 명확해야 합니다.

다음 결과가 나오면 ESNN의 practical value를 낮춰야 합니다.

1. radial/tangential transport가 isotropic baseline과 compute-matched 상태에서 거의 차이가 없다.
2. feature-conditioned matrix transport의 gain이 parameter/FLOP를 맞추면 사라진다.
3. $l=2$ carrier가 같은 wall-clock/memory에서 일관되게 더 강하다.
4. unseen rotation/reflection test에서 작은 but systematic equivariance drift가 성능과 함께 증가한다.
5. preferred-direction pathway가 물리적 axis가 아니라 dataset coordinate convention만 복구한다.
6. protein–ligand adaptation에서 ID는 좋아지지만 target/scaffold OOD가 악화된다.

반대로 위 controls를 통과하면 `operator-centric scaling`이라는 주장이 훨씬 강해집니다.

## 22. 이 논문의 가장 재사용 가능한 연구 질문

이 논문을 한 문장으로 압축하면 다음 질문입니다.

$$
\boxed{
\text{표현력을 node state의 order가 아니라 relation operator에 배치하면 무엇을 얻는가?}
}
$$

이 질문은 equivariant GNN에만 국한되지 않습니다.

- attention에서 value transform을 relation-conditioned operator로 만들기;
- pair representation을 node transport의 hypernetwork로 사용하기;
- message passing에서 isotropic kernel과 anisotropic operator를 분리하기;
- higher-order representation과 operator complexity를 동일 budget에서 trade-off하기

같은 방향으로 일반화할 수 있습니다.

따라서 ESNN은 특정 layer를 그대로 복제하기보다 **architecture capacity placement**를 다시 생각하게 만드는 논문으로 저장할 가치가 있습니다.

## 23. Final verdict

**Verdict: Must Read for equivariant architecture design, but not yet a drop-in protein–ligand solution.**

논문의 가장 강한 contribution은 first-order equivariant model의 표현력 한계를 `더 높은 l을 추가하자`로만 풀지 않고 **edge transport operator 자체를 구조화하고 학습하자**고 재정의한 점입니다.

radial/tangential decomposition은 이 아이디어를 매우 명확하게 만듭니다. edge direction 하나만 있을 때 허용되는 operator family가 무엇인지 알면, 어떤 추가 covariant feature가 실제로 새로운 표현력을 주는지도 구분할 수 있습니다.

반면 public artifact가 아직 부족하고 protein–ligand interaction benchmark가 직접 evidence에 포함된 것은 아니므로, 실무 적용에서는 `ESNN implementation 복제`보다 **transport-only ablation**부터 시작하는 것이 합리적입니다.

## 6개월 뒤 기억해야 할 세 가지

1. **ESNN의 핵심은 higher-order representation이 아니라 matrix-valued edge transport다.** Capacity를 node carrier가 아니라 relation/operator에 놓는다.
2. **relative displacement만 있으면 linear $O(n)$-equivariant transport는 radial/tangential decomposition으로 강하게 제한된다.** 추가 learned covariant feature가 richer transport를 가능하게 한다.
3. **가장 중요한 검증은 compute-matched operator-vs-representation ablation이다.** isotropic → radial/tangential → feature-conditioned transport → higher-order carrier를 symmetry/OOD/efficiency까지 같이 비교해야 한다.

## Sources

- [arXiv:2608.28853](https://arxiv.org/abs/2608.28853)
- [Author page: Alessio Borgi](https://alessioborgi.github.io/)
- [E(n) Equivariant GNN note](/papers/architectures/egnn)
- [TriELA public repository](https://github.com/eightmm/equivariant-linear-attention)
