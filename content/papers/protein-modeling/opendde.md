---
title: OpenDDE — Folding, Reasoning, and Scaling with an Open-Source Drug Discovery Engine
aliases:
  - papers/opendde
tags:
  - papers
  - protein-modeling
  - computational-biology
  - structure-prediction
  - co-folding
  - antibody-antigen
  - diffusion
  - pairformer
status: full-note
source_type: ArXiv
source_url: https://arxiv.org/abs/2607.03787
---

# OpenDDE: Folding, Reasoning, and Scaling with an Open-Source Drug Discovery Engine

> **한 줄 요약:** OpenDDE에서 가장 오래 남을 아이디어는 “drug-discovery engine”이라는 넓은 이름보다, **residue-level Pairformer와 all-atom diffusion 사이에 backbone·side-chain·base·ligand 같은 의미를 가진 structural-token relational state를 명시적으로 삽입해 coarse-to-fine geometry reasoning을 수행한다는 설계**다. 다만 현재 가장 강한 evidence는 antibody–antigen co-folding이며, 논문 자체가 ligand docking·virtual screening·affinity ranking에 대한 현재 성능 주장을 명시적으로 제한한다.

## 왜 이 리포트를 남기는가

AlphaFold3 계열 all-atom co-folding model을 볼 때 architecture 설명은 자주 다음처럼 압축된다.

```text
sequence / MSA / template / chemistry
        ↓
single + pair trunk
        ↓
all-atom diffusion
        ↓
coordinates
```

이 그림은 전체 pipeline을 이해하기에는 충분하지만, 실제 모델 설계에서는 중요한 질문 하나를 숨긴다.

> **residue-level pair state가 atom coordinate로 내려가기 전에, side-chain·base·ligand처럼 실제 interface geometry를 결정하는 substructure 사이 관계를 어디에서 reasoning할 것인가?**

OpenDDE가 흥미로운 이유는 이 질문에 매우 직접적인 interface를 제안하기 때문이다.

$$
(S^r,Z^r)
\rightarrow
(S^s,Z^s)
\rightarrow
X
$$

- $(S^r,Z^r)$: residue/molecular-unit 수준의 single/pair latent state
- $(S^s,Z^s)$: 더 세밀한 structural-token 수준 single/pair latent state
- $X$: all-atom coordinates

즉 “더 깊은 Pairformer”와 “더 강한 diffusion decoder” 사이에 **semantic resolution change**를 둔다. Protein residue 하나를 끝까지 하나의 token으로 유지하지 않고, coordinate generation 직전에 backbone·side-chain 등 실제 geometric role로 다시 펼친 뒤 pair-conditioned attention과 triangle update를 수행한다.

이 note를 저장하는 두 번째 이유는 evidence reading 때문이다. OpenDDE는 antibody–antigen benchmark에서 강한 author-reported 결과를 보이지만, 동시에 **ranked performance와 oracle performance 사이에 큰 간격**이 있다. 이는 모델이 좋은 구조를 “생성할 수 있는가”와 좋은 sample을 “선택할 수 있는가”가 다른 문제임을 잘 보여준다.

세 번째 이유는 논문이 자신의 protein–ligand claim boundary를 드물게 명확하게 적어 놓았기 때문이다. OpenDDE는 ligand/atom structural token을 architecture에 포함하지만, 현재 release가 ligand docking, virtual screening, affinity ranking에 최적화되었다고 주장하지 않는다. **representation capability와 task evidence를 분리해서 읽어야 한다.**

---

## Metadata and public artifacts

| Field | Value |
| --- | --- |
| Paper | Folding, Reasoning, and Scaling with Open-source Drug Discovery Engine |
| Authors | OpenDDE Project, Aureka AI Research |
| Version | arXiv v1 |
| Submitted | 2026-07-04 |
| arXiv | [2607.03787](https://arxiv.org/abs/2607.03787) |
| Parameters | 655M trainable parameters, author-reported |
| Residue single width $c_s$ | 384 |
| Pair width $c_z$ | 384 |
| Pairformer | 48 blocks |
| Structural-token roles | 7 |
| Structural Refiner | 4 blocks, 8 heads |
| Diffusion transformer | 24 blocks |
| Main evaluation emphasis | protein–protein and antibody–antigen co-folding |
| New benchmark | 2026ARK-AB, 164 PDB complexes / 159 interface clusters |
| Official code | [aurekaresearch/OpenDDE](https://github.com/aurekaresearch/OpenDDE) |
| Code snapshot inspected | `ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f` |
| Public release inspected | OpenDDE 1.1.1 |
| Public checkpoints | general OpenDDE + antibody–antigen checkpoint |
| Paper license | CC0 1.0 |
| Code license | Apache-2.0 |

> **Claim boundary:** benchmark 수치와 scaling 해석은 논문 저자들이 보고한 결과다. 공개 code/checkpoint의 존재와 current implementation 구조는 독립적으로 확인할 수 있지만, 성능 수치 자체를 이 note에서 재실행해 검증한 것은 아니다.

---

## Figure guide — 먼저 이 네 그림을 보면 된다

Paper는 CC0 1.0이므로 아래 scientific figures를 원문 출처와 함께 직접 보여준다. 각 figure는 **저자 보고 evidence**이며, 독립 재현 결과로 읽으면 안 된다.

### Figure 7 — OpenDDE 전체 architecture와 structural-token reasoning

![OpenDDE architecture and structural-token reasoning](https://arxiv.org/html/2607.03787v1/opendde_arch.png)

*Source: OpenDDE technical report, Figure 7, [arXiv:2607.03787](https://arxiv.org/abs/2607.03787), CC0 1.0. Panel (a)는 residue-level trunk에서 structural-token branch와 diffusion으로 내려가는 전체 흐름, (b)는 Atom37 geometry를 semantic structural tokens로 재조직하는 과정, (c)는 shape-complementarity objective의 intuition을 보여준다.*

**볼 것:** OpenDDE의 novelty를 “Pairformer를 384로 키웠다”와 “새 structural refiner를 추가했다”로 분리해서 봐야 한다. 앞쪽 trunk는 global relational context를 만들고, structural branch는 backbone/side-chain/base/ligand role을 노출한 뒤 더 세밀한 관계를 다시 reasoning한다.

---

### Figure 2 — rank와 oracle을 반드시 분리해서 읽어야 한다

![OpenDDE antibody-antigen benchmark rank and oracle](https://arxiv.org/html/2607.03787v1/x2.png)

*Source: OpenDDE technical report, Figure 2, CC0 1.0. PXMeter-AB, FoldBench-AB, 2026ARK-AB에서 top-ranked selection과 ground-truth DockQ로 best sample을 고르는 oracle selection을 나란히 비교한다.*

저자 보고 top-ranked `DockQ > 0.23` success는:

- PXMeter-AB: **51.0%**
- FoldBench-AB: **70.0%**
- 2026ARK-AB: **66.4%**

반면 oracle은 각각 **65.9%, 81.9%, 80.1%**다.

이 차이는 단순 “더 좋은 숫자”가 아니다.

$$
\text{sampling capacity}
\neq
\text{ranking quality}
$$

좋은 candidate를 생성하는 능력이 이미 있어도 confidence/ranking이 그 candidate를 top-1으로 선택하지 못하면 deployable performance는 낮게 남는다.

---

### Figure 4 — scaling law라기보다 cross-model scaling observation으로 읽기

![OpenDDE cross-model scaling plots](https://arxiv.org/html/2607.03787v1/x4.png)

*Source: OpenDDE technical report, Figure 4, CC0 1.0. 저자들은 training tokens와 `tokens × parameters` proxy에 대해 여러 co-folding model의 antibody–antigen 성능을 비교한다.*

저자들은 OpenDDE의 estimated training tokens를 약

$$
2.04\times 10^{10}
$$

으로, parameter count 655M을 곱한 training-cost proxy를 약

$$
1.33\times10^{19}
$$

으로 계산한다.

하지만 이 plot은 **동일 architecture를 여러 compute budget으로 통제한 scaling sweep가 아니다.** AlphaFold3, Protenix, SeedFold, ESMFold2, OpenDDE는 architecture, data mixture, distillation, post-training, inference가 서로 다르다.

따라서 가장 안전한 해석은:

> 큰 scale의 최신 co-folding systems가 이 benchmark에서 더 높은 성능을 보이는 **관찰적 경향**은 있지만, Figure 4 하나로 clean scaling law나 causal exponent를 확정할 수는 없다.

---

### Figure 5 — test-time compute의 병목은 sampling보다 selection일 수 있다

![OpenDDE test-time scaling](https://arxiv.org/html/2607.03787v1/x5.png)

*Source: OpenDDE technical report, Figure 5, CC0 1.0. Seed 수를 늘릴 때 ranked high-DockQ와 oracle success가 어떻게 변하는지 비교한다.*

FoldBench-AB에서 저자 보고 ranked `DockQ > 0.8` success는 약 28%에서 34% 정도로 증가하지만, oracle success는 약 66%에서 90% 이상까지 상승한다. 2026ARK-AB도 같은 패턴을 보인다.

즉 compute를 더 쓰면 sample set 안에는 훨씬 좋은 구조가 들어오지만, ranker가 그 증가분을 충분히 회수하지 못한다.

이 observation은 future architecture보다 **confidence calibration, listwise ranking, consensus selection, independent rescoring**이 큰 leverage를 가질 수 있다는 실험 가설을 만든다.

---

## 1. Problem: residue token과 atom coordinate 사이의 representation gap

Residue-level pair representation은 global context를 표현하기 좋다.

$$
S^r\in\mathbb{R}^{N_r\times c_s},
\qquad
Z^r\in\mathbb{R}^{N_r\times N_r\times c_z}.
$$

여기서 residue $i$ 하나는 backbone과 여러 side-chain atom을 포함하지만 trunk에서는 하나의 token으로 압축되어 있다.

이 압축은 효율적이다. Dense pair memory는 대략

$$
O(N_r^2c_z)
$$

이므로 처음부터 모든 atom을 dense pair token으로 쓰는 것보다 훨씬 싸다.

문제는 interface packing이 residue centroid 수준에서 끝나지 않는다는 것이다.

- backbone orientation
- side-chain rotamer
- donor/acceptor geometry
- aromatic/base orientation
- ligand atom placement
- steric exclusion

같은 항목은 더 세밀한 object 사이 관계에 의존한다.

반대로 atom-level diffusion에게 이 모든 relational reasoning을 마지막에 몰아주면 decoder가 `global relation inference`와 `coordinate denoising`을 동시에 해결해야 한다.

OpenDDE의 structural-token branch는 이 두 극단 사이에 중간 state를 만든다.

```text
coarse global reasoning
        ↓
semantic fine relational reasoning
        ↓
coordinate generation
```

이것이 이 논문의 architecture를 보는 가장 유용한 프레임이다.

---

## 2. Residue trunk: 크게 키운 Pairformer가 먼저 global hypothesis를 만든다

OpenDDE는 AlphaFold3/Protenix 계열과 유사하게 sequence, atom, MSA, template, constraint feature에서 residue/molecular-unit single/pair state를 만든다.

Main trunk는 48 Pairformer blocks이고 paper가 보고하는 width는:

$$
c_s=384,\qquad c_z=384.
$$

논문은 AlphaFold3의 pair hidden dimension 128과 비교해 이를 384로 확장했다고 설명하며, Pairformer parameter count를 대략 3배, 계산량을 대략 9배 늘리는 변화라고 적는다.

이 점은 structural-token refiner 효과를 읽을 때 중요하다.

**OpenDDE의 성능 차이를 refiner 하나의 결과로 귀속하면 안 된다.**

동시에 바뀐 것이 적지 않다.

- Pairformer width
- training scale
- data mixture
- antibody-specific late-stage weighting
- auxiliary geometry losses
- sampling/inference recipe
- structural-token branch

따라서 paper는 강한 system evidence를 제공하지만, component attribution evidence는 훨씬 약하다.

---

## 3. Structural-token expansion: semantic resolution을 바꾼다

Residue-level state를 그대로 atom별 token으로 단순 복제하는 것이 아니다. OpenDDE는 각 structural token에 **role identity**와 parent relation을 부여한다.

Paper의 개념적 mapping은:

$$
i\rightarrow\{u:\pi(u)=i\},
$$

where $\pi(u)$ is the parent residue or molecular unit.

Protein residue를 예로 들면 대략:

```text
residue token i
   ├─ backbone structural token
   └─ side-chain structural token
```

Nucleic acid에는 backbone/base role이 있고, ligand/other atom에는 별도 role이 존재한다. Paper는 총 7 structural-token roles를 보고한다.

이 expansion에서 핵심은 token 수 증가 자체가 아니다.

### 3.1 Single state의 semantic duplication

Residue state $s_i^r$에서 child token $u$의 초기 state를 개념적으로

$$
s_u^s
=
W_s s_{\pi(u)}^r
+
e_{\rho(u)}
+
\phi_u
$$

처럼 생각할 수 있다.

- $\rho(u)$: structural role
- $e_{\rho(u)}$: role embedding
- $\phi_u$: structural/atom-context feature

즉 같은 parent residue에서 나온 backbone과 side chain도 동일 representation으로 남지 않는다.

### 3.2 Pair state의 fine-grained re-indexing

Residue pair $z_{ij}^r$도 child structural token pair $(u,v)$로 확장된다.

$$
z_{uv}^s
\leftarrow
W_z z_{\pi(u)\pi(v)}^r
+
\psi(u,v).
$$

여기서 $\psi$에는 같은 residue membership, polymer adjacency, backbone–side-chain/base relation 같은 explicit relation feature가 들어갈 수 있다.

이 설계의 장점은 **dense pair reasoning을 완전히 atom level로 폭발시키지 않으면서도, residue보다 chemical meaning이 강한 intermediate pair graph를 얻는 것**이다.

---

## 4. Structural Refiner: coordinate 전에 pair geometry를 한 번 더 합의시킨다

Expansion 후 OpenDDE는 4-block, 8-head Structural Refiner를 적용한다. 공개 implementation에서도 `StructuralTokenExpander` 뒤에 별도의 `PairformerStack`이 structural refiner로 연결되어 있다.

Structural-token graph를

$$
G_s=(V_s,E_s)
$$

로 생각하면:

- node: backbone, side-chain, base, ligand/atom structural token
- edge: refined pair representation $z_{uv}^s$

가 된다.

### Pair-conditioned attention

Node update가 단순 content attention만 보지 않고 pair state를 bias/context로 사용하면 개념적으로:

$$
\alpha_{uv}
\propto
\exp
\left(
\frac{q_u^\top k_v}{\sqrt d}
+
b(z_{uv}^s)
\right).
$$

즉 `u가 v를 얼마나 볼지`가 pair relation hypothesis에 조건부가 된다.

### Triangle update

더 중요한 것은 dense pair state 자체를 third-token path를 통해 갱신할 수 있다는 점이다.

$$
z_{uv}
\leftarrow
z_{uv}
+
\sum_k
f(z_{uk},z_{kv}).
$$

예를 들어 side-chain $u$와 ligand/partner token $v$ 관계를 업데이트할 때, nearby backbone/side-chain token $k$를 거치는 두 pair의 consistency를 사용할 수 있다.

그래서 structural refiner를 일반적인 local GNN으로만 이해하면 부족하다. 이 branch의 핵심은 **fine semantic token level에서도 dense pair state와 triangle reasoning을 유지**한다는 것이다.

---

## 5. 왜 “reasoning”이라고 부르는가 — 그리고 어디까지 reasoning인가

Paper는 reasoning을 다음처럼 정의한다.

> coordinate를 생성하기 전에 molecular tokens 사이 latent relational inference를 수행하는 것.

즉 LLM-style chain-of-thought나 external symbolic reasoning을 뜻하지 않는다.

$$
\text{reasoning here}
=
\text{iterative latent relation refinement}.
$$

이 구분은 중요하다. OpenDDE의 structural-token branch가 유용한 이유는 “reasoning”이라는 label 때문이 아니라:

1. representation resolution을 명시적으로 바꾸고,
2. fine relation을 coordinate보다 먼저 표현하며,
3. pair-conditioned attention과 triangle update로 consistency를 맞춘다

는 구체적인 architecture contract 때문이다.

따라서 다른 biomolecular model에 옮길 때도 이름보다 이 interface를 가져가야 한다.

---

## 6. Atom37과 structural token은 같은 것이 아니다

Paper가 별도 section을 둔 부분이다.

Atom37은 protein residue마다 최대 37 atom slot을 갖는 **coordinate/output indexing convention**이다.

$$
X_i^{37}
=
\{x_{i,a}\}_{a=1}^{37}.
$$

Structural token은 이 atom slot 자체가 아니다.

- Atom37: 최종 atom 좌표를 어디에 담을지 정의
- structural token: 좌표를 만들기 전에 어떤 chemical role 단위로 relation을 reasoning할지 정의

즉

$$
\text{Atom37}
\neq
\text{latent structural-token state}.
$$

OpenDDE가 주장하는 bridge는 다음이다.

$$
\text{residue context}
\rightarrow
\text{structural-role context}
\rightarrow
\text{atom slots / coordinates}.
$$

이 separation은 all-atom model을 설계할 때 꽤 재사용 가치가 높다. Atom representation을 더 세밀하게 만든다고 자동으로 relational reasoning이 더 세밀해지는 것은 아니기 때문이다.

---

## 7. Shape complementarity: RMSD만으로 interface를 가르치지 않는다

Complex structure에서는 두 chain의 atoms가 native와 가까워지는 것만으로 좋은 interface가 보장되지 않는다.

- surface가 서로 등을 지고 있을 수 있고,
- gap이 너무 크거나 작을 수 있고,
- local clash가 생길 수 있으며,
- side-chain packing이 부자연스러울 수 있다.

OpenDDE는 differentiable shape-complementarity term을 추가한다.

Token center $c_u$ 주위 same-chain atom density에서 approximate surface normal $n_u$를 만들고, cross-chain pair $(u,v)$에 대해:

- 서로 바라보는가
- normal이 반대 방향인가
- gap이 적절한가
- clash가 없는가

를 곱한다.

논문의 pair score는:

$$
q_{uv}
=
f_{\mathrm{face}}
f_{\mathrm{opp}}
f_{\mathrm{gap}}
f_{\mathrm{clash}}.
$$

그리고 predicted geometry와 ground-truth geometry에서 계산한 pair/token/global score를 Huber loss로 맞춘다.

$$
\mathcal L_{\mathrm{shape}}
=
\lambda_p H(q_{uv},q^*_{uv})
+
\lambda_t H(q_u,q^*_u)
+
\lambda_g H(q_{\mathrm{global}},q^*_{\mathrm{global}}).
$$

### 이 loss에서 배울 점

이 objective를 “physics loss”라고 과장하면 안 된다. Free energy나 force field가 아니다.

하지만 coordinate error가 놓치기 쉬운 **interface-facing geometry**를 별도 supervision axis로 만든다는 점은 유용하다.

즉 objective design 관점에서는:

$$
\text{coordinate correctness}
+
\text{interface compatibility}
$$

를 분리한다.

---

## 8. Local-frame / torsion supervision: all-atom packing을 따로 잡는다

Diffusion model은 refined state에 조건부로 atom coordinates를 생성한다.

$$
\hat X_0
=
D_\theta(X_t,t\mid S^s,Z^s,\mathcal A).
$$

Global coordinate loss만 사용하면 residue 내부 orientation이나 side-chain rotamer를 세밀하게 잡기 어렵다. OpenDDE는 parent local frame $T_u$에서 predicted/true atom을 비교하는 loss를 둔다.

$$
\tilde x_a^{pred}=T_u^{-1}\hat x_a,
\qquad
\tilde x_a^{true}=T_u^{-1}x_a^*.
$$

Protein side chain은 torsion supervision도 받는다.

$$
\mathcal L_\chi
=
1-\cos(\hat\chi-\chi^*).
$$

Paper가 요약하는 geometry objective는:

$$
\mathcal L_{\mathrm{geom}}
=
\mathcal L_{\mathrm{diff}}
+
\lambda_{\mathrm{shape}}\mathcal L_{\mathrm{shape}}
+
\lambda_{\mathrm{local}}\mathcal L_{\mathrm{local}}
+
\lambda_{\mathrm{sc/base}}\mathcal L_{\mathrm{sc/base}}
+
\lambda_\chi\mathcal L_\chi.
$$

즉 “diffusion이 알아서 chemistry를 배우게 둔다”보다 여러 geometry contract를 명시적으로 넣은 system이다.

---

## 9. Prediction과 design을 같은 conditional diffusion으로 묶는다

OpenDDE는 structure prediction과 conditional design을 서로 다른 architecture로 만들지 않는다.

Atom마다:

$$
m_a^{known}\in\{0,1\},
\qquad
m_a^{target}\in\{0,1\}
$$

를 정의한다.

Full prediction에서는:

$$
m^{known}=0.
$$

Conditional design에서는 일부 chain/motif/context를 고정하고 target atom에만 noise를 준다.

$$
X_t
=
m^{known}\odot\bar X_0
+
m^{target}\odot(\bar X_0+\sigma_t\epsilon).
$$

따라서 동일 model은:

```text
no known structure
    → full structure prediction

known chain / motif / structural context
    → conditional generation
```

으로 바뀐다.

이 formulation은 architecture reuse 측면에서는 깔끔하지만, **conditional design capability가 therapeutic design utility를 자동으로 의미하지는 않는다.** Sequence recovery, foldability, binding, specificity, experimental activity는 별도 evaluation layer가 필요하다.

---

## 10. Training contract: architecture와 data curriculum을 분리해서 봐야 한다

OpenDDE는 655M parameters이며 warmup + four major stages로 학습된다.

Paper의 data curriculum intuition은:

$$
\text{precision}
\rightarrow
\text{breadth}
\rightarrow
\text{precision}.
$$

초기에는 experimental structures 비중을 높여 local geometry를 배우고, 중간에는 distilled/predicted structures로 coverage를 넓히며, 후반에는 high-quality/task-specific data와 더 큰 crop으로 돌아온다.

특히 SAbDab sampling weight는:

```text
warmup / stage I: 0%
stage II:          3%
stage III:         8%
stage IV(a):      10%
stage IV(b):      13%
```

로 증가한다.

이것은 antibody–antigen 결과를 해석할 때 중요하다.

**Architecture innovation과 antibody-focused late-stage data reweighting이 동시에 존재한다.**

따라서 “structural token refiner 때문에 Ab–Ag가 좋아졌다”는 attribution은 현재 evidence로는 성립하지 않는다.

---

## 11. Training scale: 큰 모델이지만 cost definition을 정확히 읽기

Paper는 preprocessing에 약 5주, main training에 대규모 Ampere GPU fleet 6주, 이후 Hopper에서 추가 1주를 사용했다고 보고하며 총 training compute를 약 **414K GPU-hours**라고 기술한다.

또 scaling plot에서:

$$
T_{\mathrm{train}}
=
\text{steps}\times\text{GPUs(batch)}\times\text{crop tokens}
$$

형태의 estimated training tokens를 사용한다.

이 값은 language-model token count와 같은 universal unit가 아니다. Crop size, batch semantics, structure complexity, MSA/template processing, dense pair cost가 모두 생략된 proxy다.

또

$$
C_{\mathrm{proxy}}
=
T_{\mathrm{train}}\times N_{\mathrm{params}}
$$

도 FLOPs가 아니라 비교용 proxy다.

그래서 Figure 4를 읽을 때는 절대 compute accounting보다 **model family 간 coarse scale ordering** 정도로 사용하는 것이 안전하다.

---

## 12. Evaluation boundary: 2026ARK-AB는 무엇을 테스트하는가

2026ARK-AB는 recent antibody–antigen benchmark로:

- 164 PDB complexes
- 159 unique interface clusters
- MMseqs2 entity clustering
- minimum sequence identity 40%
- alignment coverage 80%

를 사용한다고 보고한다.

Interface cluster는 interacting entity-cluster pair로 정의된다.

이 protocol은 random structure split보다 분명 더 강한 test다. 최근 공개 structure라는 temporal element도 있다.

하지만 “recent + 40% cluster”만으로 모든 leakage가 사라지는 것은 아니다.

검토해야 할 axis는 여전히 남는다.

- MSA homolog proximity
- template database relation
- antigen family similarity
- antibody germline / CDR similarity
- structural motif similarity
- benchmark construction와 model-selection interaction

따라서 저자들이 사용하는 `generalization to emerging biological systems`라는 표현은 benchmark-defined scope 안에서 읽어야 한다.

---

## 13. 결과를 읽는 핵심: Top-1과 oracle을 하나의 성능으로 합치지 않는다

Figure 2에서 가장 중요한 observation은 OpenDDE의 높은 success 자체와 함께 **oracle gap**이다.

2026ARK-AB 예를 들면:

$$
66.4\% \quad \text{ranked}
$$

vs.

$$
80.1\% \quad \text{oracle}.
$$

Oracle은 ground-truth DockQ를 사용해 후보 중 best를 고른다. 실제 deployment에서는 사용할 수 없다.

따라서:

- ranked score → 현재 usable end-to-end system quality
- oracle score → current sampler candidate set의 upper-bound capacity

로 분리해야 한다.

이 gap이 크다는 것은 “model capacity가 좋다”와 동시에 “ranking이 병목이다”라는 두 해석을 만든다.

---

## 14. Test-time scaling: sample 수 증가보다 ranker가 따라오는지가 중요하다

Seed 수를 늘릴수록 oracle curve는 크게 올라가지만 ranked curve는 완만하게 오른다.

이것은 stochastic structure model에서 흔히 중요한 pattern이다.

$$
\max_{k\le K} Q(x_k)
\uparrow
\quad\text{with }K,
$$

하지만 실제 selector $r_\phi(x_k)$가 ground-truth quality $Q$와 충분히 정렬되지 않으면:

$$
Q(x_{\arg\max r_\phi})
$$

는 훨씬 느리게 증가한다.

따라서 test-time scaling study는 단순히 “500 seeds를 쓰면 더 좋다”는 결론보다:

> **sampling budget을 늘리기 전에 ranking calibration과 independent selection objective를 개선할 정보가 충분히 있는가?**

라는 질문으로 이어지는 편이 더 decision-useful하다.

---

## 15. Reproducibility: 공개 수준은 높은 편이지만 preview라는 점을 기억해야 한다

OpenDDE의 장점 중 하나는 public artifact surface가 넓다는 점이다.

확인 가능한 공개 항목은:

- Apache-2.0 code
- training/inference implementation
- general checkpoint
- antibody–antigen checkpoint
- benchmark material
- model/training hyperparameters
- staged data mixture
- sampling configuration

이다.

Current public repository는 이 note 작성 시점에 version 1.1.1이며 structural-token expansion/refiner가 실제 model path에 포함되어 있다.

다만 repository는 preview/evolving release 성격을 명시한다. API/checkpoint behavior가 업데이트될 수 있으므로 재현 experiment는 반드시 commit/checkpoint/version을 pin해야 한다.

---

## 16. 가장 중요한 claim boundary: 이 논문은 현재 protein–ligand docking evidence가 아니다

OpenDDE는 input object로 ligand를 받을 수 있고 structural token role에도 ligand/atom이 있다. Shape-complementarity objective도 protein–ligand interface에 형식적으로 적용 가능하다.

하지만 이것만으로 ligand docking model이라고 부르면 evidence를 넘어간다.

Paper limitation은 매우 명확하다.

- 현재 study의 중심은 protein–protein / antibody–antigen
- PPI performance와 protein–ligand pose performance 사이 correlation이 제한적이었다고 저자들이 보고
- present model이 ligand docking, virtual screening, affinity ranking에 optimized되었다고 주장하지 않음
- ligand-focused data, chemistry-aware post-training, task-specific evaluation이 필요

따라서 다음 implication은 금지해야 한다.

$$
\text{strong Ab–Ag DockQ}
\not\Rightarrow
\text{strong ligand RMSD / screening / affinity}.
$$

이 boundary가 오히려 이 paper를 좋은 연구 자료로 만든다. “universal biomolecular foundation”이라는 broad narrative와 실제 validated task 사이의 거리를 저자들이 스스로 명시하고 있기 때문이다.

---

## 17. 실제 novelty는 무엇인가

OpenDDE의 contribution을 세 층으로 나누면 더 명확하다.

### A. Architecture novelty

**Structural-token expansion + fine relational refiner**

Residue-level global reasoning과 atom-level coordinate generation 사이에 semantic structural relation state를 둔다.

### B. Objective novelty

**Shape complementarity + local packing supervision**

Coordinate loss 외에 facing/gap/clash/local-frame/torsion을 별도 geometry contract로 준다.

### C. System/training novelty

**Scale + curriculum + conditional design + test-time sampling**

Pairformer width, large compute, staged data mixture, antibody weighting, unified conditional diffusion, many-seed inference가 함께 system 성능을 만든다.

가장 재사용 가치가 높은 것은 A지만, 현재 benchmark improvement는 A/B/C가 섞인 결과다.

---

## 18. 무엇이 아직 unsupported인가

현재 paper evidence만으로 다음 주장을 하면 과하다.

### “Structural Refiner가 성능 향상의 주원인이다”

Clean matched ablation이 충분하지 않다. Pairformer width와 data/training recipe가 동시에 바뀐다.

### “Figure 4가 biomolecular universal scaling law를 증명한다”

Cross-model observational comparison이다. Controlled compute sweep가 아니다.

### “OpenDDE가 drug discovery 전 과정을 해결한다”

현재 validated core는 folding/co-folding이다. Affinity, virtual screening, active learning, experimental feedback는 future extension으로 구분된다.

### “Ab–Ag 결과가 protein–ligand로 전이된다”

논문이 직접 그 extrapolation을 제한한다.

---

## 19. 실패 가능성과 confounder

### 19.1 Data mixture confound

SAbDab weight가 후반부에 크게 증가한다. Ab–Ag 성능 향상을 architecture와 분리하기 어렵다.

### 19.2 Capacity confound

$c_z=384$와 48-block Pairformer 자체가 매우 큰 change다. Structural-token branch만 비교하는 matched model이 필요하다.

### 19.3 Ranking bottleneck

Oracle gap이 커서 generation improvement가 end-to-end top-ranked utility로 모두 이어지지 않는다.

### 19.4 Structural-token compute

Residue count $N_r$보다 structural-token count $N_s$가 커지면 dense pair reasoning은 대략

$$
O(N_s^2c_z)
$$

로 비싸진다. Refiner가 얕은 4 blocks인 것은 이 representation의 benefit/cost trade-off와 연결해 볼 수 있다.

### 19.5 Generalization boundary

Recent low-homology benchmark는 강한 evidence지만 sequence clustering만으로 template/MSA/structural similarity의 모든 leakage channel을 대표하지 않는다.

### 19.6 Broad task naming

“drug discovery engine”이라는 이름이 structure prediction evidence보다 넓다. Note에서는 current validated core와 roadmap을 분리해서 본다.

---

## 20. 가장 가치 있는 ablation은 무엇인가

OpenDDE를 정말 이해하려면 leaderboard보다 아래 matched experiment가 더 중요하다.

### Experiment A — structural-token branch의 순수 효과

동일:

- training data
- crop schedule
- Pairformer width/depth
- diffusion decoder
- seed budget

에서:

```text
A0: residue trunk → diffusion
A1: residue trunk → structural expansion → diffusion
A2: residue trunk → expansion → structural refiner → diffusion
```

를 비교한다.

Primary metrics는:

- DockQ distribution
- interface clash rate
- side-chain local geometry
- confidence calibration
- compute/memory

여야 한다.

### Experiment B — Pairformer scale와 refiner를 factorize

```text
c_z=128, no refiner
c_z=384, no refiner
c_z=128, refiner
c_z=384, refiner
```

같은 factorial design이 있으면 `scale`와 `semantic refinement`를 훨씬 잘 분리할 수 있다.

### Experiment C — shape loss가 무엇을 실제로 바꾸는가

Shape loss on/off에서:

- DockQ
- interface RMSD
- clash
- surface gap
- side-chain contact recovery

를 나눠 봐야 한다.

DockQ 하나만 좋아지면 loss가 어떤 geometry failure를 줄였는지 알기 어렵다.

### Experiment D — sampler와 ranker 분리

$K$ seeds에 대해:

1. default confidence rank
2. independent rescoring
3. consensus/cluster rank
4. learned listwise rank
5. oracle

를 같은 candidate pool에서 비교한다.

Oracle gap을 줄이는 것이 새 sampler를 만드는 것보다 큰 leverage인지 바로 알 수 있다.

---

## 21. Protein–ligand에 가져가려면 어떤 실험이 필요한가

OpenDDE paper의 PL boundary를 존중하면서 architecture idea만 테스트하려면 matched transfer experiment가 적절하다.

```text
residue/token pair trunk
        ↓
[control] direct atom decoder
        vs
[variant] chemistry-semantic structural-token refiner
        ↓
same coordinate generator
```

Protein–ligand에서는 structural token의 의미도 다시 설계해야 한다.

단순 `ligand atom` 하나의 role만 두는 것보다:

- ligand atom identity
- bond/topology
- protein side-chain
- metal/ion
- local interaction motif

가 fine relation state에서 어떻게 만나야 하는지를 명시해야 한다.

평가도 Ab–Ag DockQ 대신 task에 맞춰 바꿔야 한다.

- pose RMSD / symmetry-aware RMSD
- PoseBusters-style chemical validity
- clash / strain
- ligand/scaffold split
- protein-family split
- temporal split
- apo/holo or pocket-state shift
- independent scoring / screening metric

즉 **architecture transfer는 가능하지만 evidence transfer는 불가능하다.**

---

## 22. 다른 co-folding model과 비교할 때 보는 축

OpenDDE를 AlphaFold3, Protenix, SeedFold 같은 모델과 비교할 때 단순 “triangle update를 쓰는가”보다 다음 질문이 더 좋다.

| Axis | Question |
| --- | --- |
| Coarse state | residue/token single-pair state가 얼마나 오래 유지되는가? |
| Resolution transition | atom/chemical subunit로 언제 내려가는가? |
| Fine relation | coordinate 전에 fine pair state가 존재하는가? |
| Triangle reasoning | 어느 resolution에서 몇 번 수행되는가? |
| Geometry objective | distance/RMSD 외 interface-specific loss가 있는가? |
| Decoder feedback | fine geometry가 coarse pair state로 다시 올라가는가? |
| Ranking | sample quality와 confidence selection을 어떻게 연결하는가? |
| Compute | dense pair width/depth와 fine-token expansion cost는 얼마인가? |

OpenDDE는 특히 **resolution transition 이후에도 pair reasoning을 유지한다**는 점에서 비교 가치가 있다.

---

## 23. 한 가지 아쉬운 점: fine reasoning은 mostly one-way다

OpenDDE의 conceptual path는:

$$
(S^r,Z^r)
\rightarrow
(S^s,Z^s)
\rightarrow
X.
$$

이것은 명확하고 구현하기 좋다.

하지만 더 강한 coarse-to-fine reasoning을 생각하면 다음 feedback도 가능하다.

$$
(S^r,Z^r)
\leftrightarrow
(S^s,Z^s)
\rightarrow
X.
$$

즉 fine structural-token branch에서 발견한 local incompatibility를 coarse residue-level pair state로 다시 올려 보내는 recycle/mixer다.

현재 OpenDDE의 핵심 contribution은 fine branch를 추가한 것이고, **bidirectional multi-resolution relational reasoning**은 자연스러운 다음 research question이다.

이 아이디어는 paper가 입증한 결과가 아니라 이 note의 확장 해석이다.

---

## 24. 구현 관점에서 확인한 것

Current public OpenDDE code snapshot에서는 main model이 실제로:

- `StructuralTokenExpander`
- optional structural token refiner
- structural pair attention bias
- Pairformer stack
- shape-complementarity utilities
- diffusion module

를 연결한다.

즉 technical report의 structural-token description이 단순 conceptual figure에만 있는 것은 아니다.

다만 current repository는 paper 이후 계속 업데이트된 public release다. 그래서 paper v1의 exact training checkpoint와 current source tree를 동일한 code state로 간주하면 안 된다.

재현 시에는 반드시:

```text
paper version
+ code commit
+ package version
+ checkpoint identity
+ inference config
```

를 함께 기록해야 한다.

---

## 25. Falsification: 어떤 결과가 나오면 structural-token story를 약하게 봐야 하나

좋은 architecture story는 반증 조건이 있어야 한다.

다음 결과가 나오면 OpenDDE의 structural-token mechanism에 대한 강한 해석은 약해진다.

1. matched data/compute에서 refiner on/off 차이가 거의 없다.
2. $c_z$ 증가만으로 같은 성능 개선이 재현된다.
3. shape loss가 DockQ는 올리지만 clash/local packing은 개선하지 않는다.
4. antibody data reweighting만으로 대부분의 gain이 설명된다.
5. independent benchmark에서 oracle gain은 유지되지만 top-ranked gain이 사라진다.
6. PL task에서 fine structural state가 direct atom decoder보다 나아지지 않는다.

반대로 위 controls를 통과한다면 structural-token refiner는 꽤 강한 reusable architecture primitive가 된다.

---

## 26. Reproducibility checklist

| Item | Status |
| --- | --- |
| Paper public | yes |
| Paper license | CC0 |
| Code public | yes, Apache-2.0 |
| General checkpoint | yes |
| Ab–Ag checkpoint | yes |
| Benchmark release | yes |
| Main model hyperparameters | reported |
| Data-stage mixture | reported |
| Training compute | reported |
| Inference sample steps | reported |
| Exact component attribution | limited |
| Independent reproduction in this note | no |
| Current PL docking validation | explicitly not established |

---

## 27. Related notes

이 paper는 다음 note들과 같이 읽는 것이 좋다.

- [[papers/architectures/alphafold3|AlphaFold3]] — residue/token Pairformer에서 all-atom diffusion으로 가는 기본 co-folding reference
- [[papers/protein-modeling/backflip-2|BackFlip-2]] — static structure에서 directional dynamics descriptor를 예측하는 다른 종류의 structural representation
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/sbdd/protein-ligand-split|Protein–ligand split]]

특히 AlphaFold3와 비교할 때 “누가 더 좋다”보다 **coarse pair → fine structural state → coordinate**라는 intermediate interface가 새로 무엇을 표현하는지 보는 편이 유용하다.

---

## Final verdict

OpenDDE를 “open AlphaFold3 clone with bigger compute”로만 읽으면 structural-token branch를 놓친다. 반대로 “atomic reasoning이 Ab–Ag 성능을 만들었다”고 읽으면 current evidence를 과대해석한다.

이 technical report에서 가장 durable한 메시지는 다음이다.

> **Dense relational state를 residue level에서 끝내지 말고, coordinate generation 직전에 chemically meaningful substructure resolution로 다시 펼친 뒤 관계를 한 번 더 reasoning하라.**

그 다음으로 중요한 lesson은 sampling과 ranking의 분리다. OpenDDE의 큰 oracle gap은 diffusion sampler가 좋은 candidate를 만들 수 있는 것과 deployable selector가 그 candidate를 찾는 것이 별개의 optimization problem임을 보여준다.

마지막으로 protein–ligand 연구자에게 가장 중요한 것은 negative lesson이다. **all-atom universal architecture와 ligand token을 가지고 있어도, ligand pose/ranking evidence가 없으면 PL capability를 주장하면 안 된다.** OpenDDE authors도 이 선을 명시적으로 긋는다.

---

## Three durable takeaways

1. **Representation resolution은 architecture primitive다.**  
   Residue pair state에서 atom coordinate로 곧바로 내려가기보다 structural-role single/pair state를 두는 것은 global context와 local chemistry 사이의 명시적 bridge가 된다.

2. **Sample generation과 sample selection을 따로 평가해야 한다.**  
   Test-time scaling에서 oracle이 ranked performance보다 훨씬 빠르게 올라간다면 다음 bottleneck은 sampler가 아니라 confidence/ranker일 수 있다.

3. **Broad biomolecular architecture claim과 task-specific evidence를 분리하라.**  
   OpenDDE의 현재 가장 강한 evidence는 antibody–antigen co-folding이다. Protein–ligand docking, screening, affinity에는 별도 data/post-training/evaluation이 필요하며, architecture transfer와 evidence transfer는 같은 것이 아니다.

---

## Sources

- OpenDDE Project, Aureka AI Research. [Folding, Reasoning, and Scaling with Open-source Drug Discovery Engine](https://arxiv.org/abs/2607.03787), arXiv:2607.03787v1, 2026.
- Official implementation: [aurekaresearch/OpenDDE](https://github.com/aurekaresearch/OpenDDE).
- Current public code snapshot inspected: [`ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f`](https://github.com/aurekaresearch/OpenDDE/commit/ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f).
- Paper figures reproduced from the official arXiv HTML under the paper's CC0 1.0 license.
