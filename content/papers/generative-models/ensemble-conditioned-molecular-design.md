---
title: Ensemble-Conditioned Molecular Design — Composable Multi-State 3D Molecular Generation
aliases:
  - papers/ensemble-conditioned-molecular-design
  - papers/ensemble-conditioned-design
tags:
  - papers
  - generative-models
  - flow-matching
  - molecular-generation
  - conditional-generation
  - structure-based-modeling
  - protein-ligand
  - conformational-ensemble
status: full-note
source_type: ArXiv
source_url: https://arxiv.org/abs/2609.15077
---

# Ensemble-Conditioned Molecular Design: Composable Multi-State 3D Molecular Generation

> **한 줄 요약:** 이 논문의 핵심은 multi-state molecule을 따로 학습하는 것이 아니라, **single-condition으로 학습된 joint continuous–discrete Flow Matching model의 조건별 endpoint/vector field를 inference에서 합성**해 여러 shape·pharmacophore·pocket state를 동시에 target하거나 일부 state를 음의 weight로 피하도록 만드는 것입니다. SBDD 관점에서 가장 재사용 가치가 높은 부분은 `target state + avoid state`를 post-hoc filter가 아니라 **generation dynamics 자체의 composable control**로 만드는 설계입니다.

## 왜 이 논문을 저장하는가

Structure-based molecular generation은 보통 하나의 pocket, 하나의 reference pose, 또는 하나의 pharmacophore pattern을 조건으로 둡니다.

하지만 실제 drug design objective는 자주 하나의 구조로 끝나지 않습니다.

- 원하는 target pocket에는 잘 맞아야 하지만 close off-target에는 덜 맞아야 합니다.
- GPCR agonist라면 active state를 선호하고 inactive state는 피해야 할 수 있습니다.
- dual-target molecule이라면 서로 다른 두 pocket에서 각각 가능한 binding mode를 가져야 합니다.
- molecule의 permeability나 flexibility는 단일 conformer가 아니라 conformational ensemble의 통계량과 연결될 수 있습니다.

즉 실제 objective는 흔히

$$
\text{one desired state}
$$

가 아니라

$$
\text{desired states}
+
\text{undesired states}
+
\text{ensemble-level properties}
$$

입니다.

가장 단순한 해법은 target/off-target 또는 active/inactive pair별 joint dataset을 만들고 그 조합마다 generator를 학습하는 것입니다. 문제는 matched multi-state data가 희소하고, 새로운 state 조합이 생길 때마다 다시 학습해야 한다는 점입니다.

이 논문은 학습과 조합을 분리합니다.

$$
\boxed{
\text{single-condition training}
\quad\rightarrow\quad
\text{multi-condition composition at inference}
}
$$

따라서 이 논문에서 오래 남는 아이디어는 “ensemble이라는 단어” 자체보다 **conditional generator를 modular controller처럼 합성하는 방법**입니다.

이 점은 [[papers/generative-models/flow-matching-rl|reward-weighted Flow Matching post-training]]과도 대비됩니다. Reward post-training은 pretrained distribution 자체를 새 objective 쪽으로 이동시키는 반면, Ensemble-Conditioned Molecular Design은 **checkpoint를 다시 바꾸지 않고 sampling-time composition으로 새 target/avoid 조합을 표현**합니다.

---

## Metadata and artifacts

| Field | Value |
| --- | --- |
| Paper | Ensemble-Conditioned Molecular Design |
| Authors | Ross Irwin, Alessandro Tibo, Jon Paul Janet, Simon Olsson |
| Version | arXiv v1 |
| Submitted | 2026-09-14 |
| arXiv | [2609.15077](https://arxiv.org/abs/2609.15077) |
| Main model | joint continuous–discrete ensemble-conditioned Flow Matching |
| Molecular state | heavy-atom coordinates + atom/charge token + bond matrix |
| Main conditions | shape, pharmacophore, protein pocket, ensemble PSA, ensemble pairwise RMSD |
| Training data | GEOM Drugs + SPINDR |
| Sampling | 100 integration steps unless otherwise stated |
| Official code | [rssrwn/ensemble-cond-design](https://github.com/rssrwn/ensemble-cond-design) |
| Code snapshot inspected | `5d557afd61875182d30e4cdfb121951ad7c0fd1c` |
| Public artifacts | pretrained checkpoint + processed GEOM/SPINDR + multi-condition benchmark on [Zenodo record 22485204](https://zenodo.org/records/22485204) |
| Paper license | CC BY 4.0 |
| Code license | MIT |

> **Claim boundary:** 아래 benchmark와 case-study 수치는 저자 보고 결과입니다. 특히 AChE/MAO-B와 A2A receptor 사례의 endpoint는 AutoDock Vina 기반이므로 **experimental affinity, selectivity, efficacy를 입증하지 않습니다.**

---

## Figure guide — 이 네 그림으로 논문 구조를 먼저 잡기

Paper가 CC BY 4.0이므로 재사용은 가능하지만, 이 note에서는 원본 scientific figure를 변형하거나 별도 asset으로 재배포하지 않고 **공식 arXiv HTML의 figure 위치에 직접 연결**합니다. 수치와 plot은 모두 저자 보고 evidence로 읽어야 합니다.

### Figure 1 — framework 전체: mode와 property를 분리한다

[Official paper — Figure 1](https://arxiv.org/html/2609.15077v1#S1.F1)

**볼 것:** molecule ensemble을 두 축으로 나눕니다. `mode`는 특정 shape/pharmacophore/pocket-compatible state이고, `property`는 ensemble 전체에서 계산되는 scalar입니다. 여러 mode에 weight $\alpha_k$를 주고 guidance strength $\gamma$로 sampling dynamics를 조합하는 것이 논문의 전체 contract입니다.

이 figure의 핵심은 조건의 종류보다 **여러 mode를 joint label로 학습하지 않아도 inference에서 조합한다**는 점입니다.

### Figure 2 — encoder–decoder architecture와 condition routing

[Official paper — Figure 2](https://arxiv.org/html/2609.15077v1#S3.F2)

**볼 것:** shape/pharmacophore profile encoder와 pocket encoder가 분리되어 있고, ensemble property는 별도 embedding을 통해 AdaLN으로 들어갑니다. Decoder는 transformer와 graph-transformer layer를 섞어 coordinates와 atom/bond endpoint를 함께 예측합니다.

여기서 architectural novelty보다 중요한 것은 **condition을 먼저 encode하고, molecule state update에서 여러 conditional prediction을 다시 호출해 합성할 수 있게 interface를 설계했다**는 점입니다.

### Figure 6 — dual AChE/MAO-B 조건이 실제로 무엇을 개선했는가

[Official paper — Figure 6](https://arxiv.org/html/2609.15077v1#S4.F6)

**볼 것:** donepezil condition 또는 safinamide condition 하나만 사용하는 control과 두 condition을 동시에 사용하는 joint generation을 비교합니다. Figure의 개선은 “dual binder가 실험적으로 나왔다”가 아니라 **동일한 Vina oracle 아래 두 reference cutoff를 동시에 만족하는 generated fraction이 증가했다**는 evidence입니다.

### Figure 7 — negative condition을 active/inactive state design에 쓰는 방식

[Official paper — Figure 7](https://arxiv.org/html/2609.15077v1#S4.F7)

**볼 것:** active A2A pocket만 condition으로 준 baseline과, inactive pocket을 negative condition으로 추가한 run을 비교합니다. 점이 diagonal 위로 이동하는 것은 active-state Vina preference가 커졌다는 뜻이지 receptor activation을 실험적으로 증명한 것은 아닙니다.

---

## 1. Problem: multi-state design은 왜 single-pocket generation과 다른가

Single-condition generator는 보통

$$
p_\theta(x\mid c)
$$

를 학습합니다. 여기서 $c$는 pocket, scaffold, shape, property 같은 하나의 condition입니다.

하지만 selectivity 문제를 생각하면 desired distribution은 자연스럽게 두 조건 사이의 관계로 정의됩니다.

예를 들어 target pocket $P_t$와 off-target pocket $P_o$가 있을 때 원하는 것은 단순히

$$
x\sim p(x\mid P_t)
$$

가 아니라 개념적으로

$$
\text{fit}(x,P_t)\uparrow,
\qquad
\text{fit}(x,P_o)\downarrow
$$

입니다.

Dual-target design은 반대입니다.

$$
\text{fit}(x,P_1)\uparrow,
\qquad
\text{fit}(x,P_2)\uparrow.
$$

이 차이는 단순 score aggregation 문제가 아닙니다. 두 pocket이나 conformer가 서로 다른 coordinate frame에 있을 수 있고, 동일 molecule graph가 각 state에서 다른 3D conformation을 가져야 할 수 있기 때문입니다.

따라서 multi-state generator에는 최소 세 가지 문제가 생깁니다.

1. **Data problem** — 같은 ligand가 여러 state/pocket에 대해 matched observation을 가진 data가 적습니다.
2. **Composition problem** — 서로 다른 conditional vector field를 어떤 규칙으로 합칠지 정해야 합니다.
3. **Reference-frame problem** — 서로 다른 coordinate frame에서 나온 equivariant feature를 그대로 더하면 geometry가 모순될 수 있습니다.

이 논문은 세 문제를 각각

- single-mode training,
- endpoint/vector-field composition,
- adaptive symmetry learning

으로 처리합니다.

---

## 2. Minimum background: endpoint-parameterized Flow Matching

Continuous Flow Matching은 simple prior $p_0$에서 data distribution $p_1$로 이동하는 time-dependent field를 학습합니다.

$$
\frac{dx_t}{dt}=v_\theta(x_t,t).
$$

이 논문은 velocity 자체보다 final endpoint를 예측하는 parameterization을 사용합니다.

$$
\hat{x}_1=\hat{x}_\theta(x_t,t).
$$

Linear path intuition에서는 velocity를

$$
v_\theta(x_t,t)
=
\frac{\hat{x}_1-x_t}{1-t}
$$

처럼 endpoint prediction으로부터 복원할 수 있습니다.

이 선택이 multi-condition composition에서 중요합니다. Endpoint prediction이 affine combination을 허용하므로 여러 condition의 prediction을 선형적으로 조합한 뒤 같은 방식으로 velocity를 얻을 수 있기 때문입니다.

### Discrete state도 같이 흐른다

Molecule은 좌표만으로 정의되지 않습니다. 논문은

$$
x=(X,a,b)
$$

를 사용합니다.

- $X\in\mathbb{R}^{N\times3}$: heavy-atom coordinates
- $a$: element + formal charge categorical token
- $b$: pairwise bond categorical matrix

Coordinates는 continuous flow, atom/bond는 discrete flow로 학습합니다. Discrete channel은 endpoint categorical distribution을 예측하고 CTMC transition rate로 sampling합니다.

따라서 multi-condition guidance도 두 형태를 동시에 맞춰야 합니다.

- coordinates: **linear combination**
- atom/bond probabilities: **log-linear / product-of-experts-like combination**

이 mixed-state contract가 이 방법을 단순한 coordinate CFG보다 더 흥미롭게 만듭니다.

---

## 3. Condition representation: mode와 ensemble property를 분리한다

논문은 condition을 크게 `mode`와 `property`로 나눕니다.

### 3.1 Shape mode

Reference conformer 좌표를 그대로 condition으로 주지 않습니다. 각 atom coordinate를 확률적으로 duplicate하고 isotropic Gaussian noise를 더해 noisy point cloud를 만듭니다.

$$
\tilde X = \operatorname{Noise}(X;\sigma_{shape}).
$$

Training에서 $\sigma_{shape}$를 $0.1$–$1.0$ 범위에서 sample하므로 inference에서 condition fidelity를 조절할 수 있습니다.

이것은 reference atom count를 그대로 leak시키지 않고 “대략 어떤 volume/shape를 원한다”는 soft geometric condition을 만드는 선택입니다.

### 3.2 Pharmacophore mode

Pharmacophore는 3D point, interaction type, 일부 direction vector로 표현합니다.

지원되는 주요 group은

- H-bond donor / acceptor,
- cation / anion,
- aromatic,
- hydrophobe

입니다.

Direction은 donor와 aromatic ring처럼 방향성이 직접 의미가 있는 일부 group에서만 사용됩니다.

SBDD 관점에서 중요한 점은 pocket 자체와 ligand-side interaction profile이 서로 다른 condition source라는 것입니다. 같은 reference frame을 공유하면 둘을 하나의 mode embedding으로 묶어 forward pass를 줄일 수 있습니다.

### 3.3 Pocket mode

Pocket condition은 reference ligand 기준 6 Å 이내 pocket residue의 atom type과 coordinates를 제공합니다.

이 condition은 exact bound pose를 복제하는 hard constraint보다 **해당 site에 맞는 pose를 model이 선택하도록 하는 soft structural condition**에 가깝습니다.

따라서 pocket-conditioned 결과와 explicit shape/pharmacophore-conditioned 결과를 같은 claim으로 읽으면 안 됩니다.

### 3.4 Ensemble property

논문은 proof-of-concept로 두 property를 사용합니다.

1. mean 3D polar surface area,
2. conformer ensemble의 mean pairwise RMSD.

이 값들은 특정 conformer가 아니라 sampled ensemble에 대해 계산됩니다.

$$
\eta
=
\bigl[
\mathbb E_{c\in\mathcal E(x)}\operatorname{PSA}(c),
\;\operatorname{MeanPairRMSD}(\mathcal E(x))
\bigr].
$$

여기서 반드시 조심해야 할 점은 paper가 실제 thermalized Boltzmann sample을 갖고 있는 것이 아니라는 것입니다. Training ensemble은 CREST 기반이고, evaluation은 비용 때문에 ETKDG + MMFF94 approximation을 사용합니다. 저자들도 따라서 thermal distribution claim보다 **conformational ensemble heuristic**으로 제한해 표현합니다.

---

## 4. Architecture: 조건은 encode하고, 생성 state는 joint decoder가 업데이트한다

Model은 크게 condition encoder와 molecular decoder로 나뉩니다.

```text
shape + pharmacophore ── profile encoder ─┐
                                          │
protein pocket ───────── pocket encoder ──┼─ cross-attention ─┐
                                          │                   │
ensemble property ────── property MLP ─── AdaLN ────────────┼─ molecular decoder
                                                              │
(x_t, atom_t, bond_t) ────────────────────────────────────────┘
                                                              ↓
                                              (X_hat1, a_hat1, b_hat1)
```

Profile encoder와 pocket encoder는 8-layer, hidden dimension 128의 transformer이고, generator는 hidden dimension 384의 16-block stack입니다. 전체 learnable parameter는 paper 기준 약 58.6M입니다.

Decoder block은 transformer와 graph-transformer를 3:1 비율로 섞습니다.

- Transformer layer: condition embedding에 cross-attention
- Graph-transformer layer: 현재 partially denoised molecule의 node/pair state 업데이트
- AdaLN: flow time과 ensemble-property condition 전달

이 구조의 중요한 engineering point는 condition encoder와 generator 역할이 분리되어 있다는 점입니다. 여러 condition을 sampling에서 조합할 때 condition을 mode별로 encode하고 같은 molecular state에 대해 conditional endpoint를 반복 계산할 수 있습니다.

### Flexible-size generation

기존 3D flow/diffusion model은 sampling 전에 atom count를 정하는 경우가 많습니다. 이 논문은 최대 48개의 atom slot을 두고 padding token 자체를 생성하도록 학습합니다.

두 trick이 핵심입니다.

1. pad atom coordinate를 molecule center-of-mass에 둡니다.
2. prior와 data의 padded atoms 사이에 permutation alignment를 적용합니다.

Ablation에서는 둘을 함께 쓴 model의 unconditional mean size가 22.1 heavy atoms이고 training mean은 24.8입니다. Permutation alignment를 제거하면 16.3, center-of-mass padding을 제거하면 18.7로 더 크게 undershoot합니다.

즉 padding은 단순 implementation detail이 아니라 **size distribution 학습을 결정하는 path design**입니다.

---

## 5. 핵심 수식: multi-condition endpoint composition

Single-condition CFG-style coordinate endpoint는

$$
\hat X^\gamma
=
(1-\gamma)\hat X_{\varnothing}
+
\gamma\hat X_{c}
$$

로 쓸 수 있습니다.

이 논문은 이를 $K$개의 mode condition으로 일반화합니다.

$$
\boxed{
\hat X^{\boldsymbol\gamma}
=
\left(1-\sum_{k=1}^{K}\gamma_k\right)
\hat X_{\varnothing}
+
\sum_{k=1}^{K}
\gamma_k\hat X_{m_k}
}
$$

Discrete atom/bond endpoint는 log probability space에서 같은 weight 구조를 사용합니다.

$$
\boxed{
\log p^{\boldsymbol\gamma}(z_1)
=
\left(1-\sum_k\gamma_k\right)
\log p_{\varnothing}(z_1)
+
\sum_k\gamma_k\log p_{m_k}(z_1)
}
$$

where $z\in\{a,b\}$.

실험에서는 보통

$$
\gamma_k=\gamma\alpha_k,
\qquad
\sum_k\alpha_k=1
$$

로 분리합니다.

- $\gamma$: 전체 guidance strength
- $\alpha_k$: condition 사이의 allocation
- $\alpha_k<0$: 해당 state를 피하는 negative design

이 formulation은 SBDD에서 직관적입니다.

### Target + target

$$
\alpha=(0.5,0.5)
$$

처럼 두 positive condition을 주면 dual-target 방향입니다.

### Target + avoid

$$
\alpha=(2,-1)
$$

처럼 한 condition에 negative weight를 주면 target state에서 멀어지지 않으면서 undesired state를 밀어내는 방향의 extrapolation을 만듭니다.

여기서 중요한 claim boundary가 있습니다. 이 수식은 각 condition을 만족하는 **정확한 joint probability distribution을 계산한다는 보장**이 아닙니다. Paper도 compositional diffusion의 vector-field composition과 product-of-experts 관점에 연결하지만, approximate composition이고 실제 behavior는 benchmark로 확인해야 합니다.

---

## 6. Adaptive symmetry learning: 서로 다른 coordinate frame을 어떻게 합치는가

Multi-pocket composition에서 가장 흥미로운 architecture choice입니다.

Suppose condition $m_1$은 coordinate frame $A$, $m_2$는 frame $B$에 있습니다. 두 encoder가 모두 frame-aware equivariant feature를 만들고 이를 그대로 합치면 서로 다른 frame의 vector information이 충돌할 수 있습니다.

Paper의 해법은 **하나의 condition만 generation frame을 정의하고, 나머지는 invariant하게 읽는 것**입니다.

Training에서 encoder input을 50% 확률로 random rotation하고, rotation 여부 flag를 AdaLN condition으로 줍니다.

Model이 학습해야 할 behavior는 대략 다음 두 가지입니다.

Invariant mode:

$$
\hat X_\theta(x_t,t,Rm,\eta)
\approx
\hat X_\theta(x_t,t,m,\eta).
$$

Equivariant mode:

$$
\hat X_\theta(Rx_t,t,Rm,\eta)
\approx
R\hat X_\theta(x_t,t,m,\eta).
$$

즉 같은 encoder가 flag에 따라 “이 condition의 orientation을 보존해야 하는가, 버려야 하는가”를 학습합니다.

### 왜 완전한 exact equivariance와 다른가

이 symmetry는 architecture 수준에서 강제로 보장하지 않습니다. Data augmentation과 flag를 통해 **behavioral symmetry를 학습**합니다.

Paper의 symmetry test에서는 trajectory 중간에 error가 존재하지만 $t\to1$에서 줄어드는 현상을 보고합니다. 저자들은 iterative denoising에서는 strict architectural symmetry가 항상 필수는 아닐 수 있다고 해석합니다.

하지만 여기에는 중요한 caveat가 있습니다.

Multi-mode benchmark에서 equivariant condition이 invariant condition보다 더 강하게 작동합니다. 예를 들어 $\sigma_{shape}=0.2$에서 어느 mode를 equivariant로 주느냐에 따라 mean shape-matching gain의 강한 축이 뒤집힙니다.

따라서 실제 사용자는

$$
\text{which condition owns the frame?}
$$

를 hyperparameter처럼 선택해야 합니다. 이 asymmetry는 단순 구현 문제가 아니라 method behavior의 일부입니다.

---

## 7. Training contract

Training objective는 coordinate MSE와 atom/bond cross entropy를 합칩니다.

$$
\mathcal L
=
\mathbb E
\left[
\omega(t)
\left(
\|\hat X_1-X_1\|^2
+
\lambda_a\operatorname{CE}(\hat a_1,a_1)
+
\lambda_b\operatorname{CE}(\hat b_1,b_1)
\right)
\right].
$$

Paper setting은

$$
\omega(t)=\min\left(\frac{t}{1-t},10\right),
\qquad
\lambda_a=0.3,
\qquad
\lambda_b=5.0.
$$

Time은 uniform이 아니라

$$
t\sim\operatorname{Beta}(1.5,1.0)
$$

에서 sample해 late-time prediction을 더 자주 봅니다.

Condition masking은 inference-time composition 가능성을 만드는 핵심입니다.

- shape profile: 30% independent mask
- pharmacophore profile: 30%
- PSA: 30%
- pairwise RMSD: 30%
- 모든 signal + pocket: 10% global mask

즉 model이 특정 fixed condition tuple에만 의존하지 않고, subset condition과 null prediction을 모두 수행하도록 학습합니다.

### Data

두 data source가 서로 다른 역할을 합니다.

**GEOM Drugs**

- 약 300K molecules
- CREST conformer ensembles
- ligand-side shape/pharmacophore/ensemble-property supervision

**SPINDR**

- 원래 약 35K protein–ligand complexes
- QED < 0.3 filter 후 training에서 약 28K systems 사용
- pocket + protein–ligand interaction supervision
- training에서 8× replication per epoch

Paper는 200 epochs, mixed-precision bf16, Adam, single A100에서 약 2일을 보고합니다.

이 training design의 핵심은 paired target/off-target data가 아니라 **single-condition coverage를 넓게 학습하고 composition은 inference로 미룬다**는 것입니다.

---

## 8. Data split과 leakage boundary

이 paper는 일반 generative benchmark보다 split을 꽤 의식적으로 구성합니다. 다만 각 split이 지지하는 generalization claim을 분리해서 읽어야 합니다.

### GEOM Drugs: unique-scaffold test

저자들은 전체 data에서 Murcko scaffold가 unique한 molecule pool을 만들고,

- heavy atoms 16–35,
- CLogP ≤ 5,
- CREST conformer ≥ 10

조건을 만족하는 후보 중 1,000 molecules를 test로 sample합니다. 그 뒤 10K validation을 떼고 나머지를 training에 사용합니다.

이것은 random molecule split보다 강하지만, **protein/pocket OOD evidence는 아닙니다.** Ligand chemical scaffold generalization과 ensemble conditioning을 보는 split입니다.

### SPINDR: PLINDER-derived protein–ligand split

Pocket-conditioned data는 SPINDR가 사용한 PLINDER-derived split을 그대로 사용합니다. Paper는 pocket/ligand similarity gap을 만들도록 train/test systems를 제거한 split이라고 설명합니다.

따라서 pocket-conditioned claim에는 random complex split보다 나은 leakage control이 있습니다. 그러나 이 note에서 이를 “새 protein family 전체에 대한 prospective generalization”으로 확장하지 않습니다. 정확한 OOD axis는 PLINDER split contract와 SPINDR preprocessing에 의해 결정됩니다.

### Multi-mode benchmark construction의 특이점

Multi-mode benchmark는 단순히 random pair를 만들지 않습니다.

- GEOM test molecule들의 compact/extended conformer를 사용
- source molecules가 서로 다른 cross-molecule pair
- target과 heavy-atom count 차이를 제한
- training molecule 중 조건을 만족할 수 있는 chemistry가 실제로 존재하는지 먼저 확인
- feasibility witness는 source molecules와 ECFP Tanimoto ≤ 0.5 조건을 둠

Mode-targeting pair 496개, mode-avoidance pair 535개를 구성합니다.

이 benchmark는 **불가능한 condition pair 때문에 method가 실패하는 confounder**를 줄인다는 장점이 있습니다. 반면 training set witness를 사용해 benchmark feasibility를 고르므로 완전히 model-agnostic한 자연 분포 benchmark는 아닙니다. “조건을 만족할 chemistry가 알려진 controlled test”에 가깝습니다.

---

## 9. Ensemble evaluation: training oracle과 test oracle이 다르다

Training ensemble은 CREST 기반이지만, generated molecule마다 CREST를 돌리면 너무 비쌉니다. 그래서 evaluation에서는 빠른 approximation을 사용합니다.

```text
SMILES / generated molecule
    ↓
ETKDG conformers (128)
    ↓
MMFF94 minimization
    ↓
RMSD 0.5 Å dedup
    ↓
> 6 kcal/mol above minimum 제거
    ↓
approximate ensemble properties
```

Validation molecule 499개에서 저자들은

- mean PSA는 CREST와 거의 동일한 수준,
- mean pairwise RMSD는 Pearson $R=0.78$

이라고 보고합니다.

여기서 중요한 것은 $R=0.78$이 “ensemble이 동일하다”는 뜻이 아니라는 점입니다. 특히 flexibility target처럼 conformer distribution tail에 민감한 objective에서는 evaluation approximation이 method ranking에 영향을 줄 수 있습니다.

따라서 이 논문의 property-control evidence는

$$
\text{CREST-trained condition}
\rightarrow
\text{ETKDG/MMFF-evaluated proxy}
$$

라는 oracle mismatch를 갖습니다.

---

## 10. Result 1: multi-mode shape targeting과 avoidance는 실제로 작동하는가

가장 controlled한 benchmark는 두 shape mode만 사용합니다.

Mode targeting에서는 두 조건 모두 positive weight를 줍니다.

$$
\alpha=(0.5,0.5).
$$

Mode avoidance에서는 하나를 target하고 다른 하나를 negative로 둡니다.

$$
\alpha=(1.5,-0.5).
$$

Metric은 raw shape similarity가 아니라 size-matched virtual-screening baseline 대비 gain입니다.

$$
\Delta
=
T_{shape}(x,m)
-
\mathbb E_{x'\sim\text{size-matched train pool}}
T_{shape}(x',m).
$$

이 설계는 큰 molecule이 단순히 volume overlap으로 유리해지는 confounder를 어느 정도 줄입니다.

Paper는 target과 avoid 모두 baseline에서 원하는 방향으로 distribution이 이동한다고 보고합니다. 동시에 $\sigma_{shape}$가 작아 condition이 더 정밀할수록 control은 강하지만 reference chemistry similarity도 커지는 trade-off가 나타납니다.

가장 눈에 띄는 failure signal은 앞서 말한 frame ownership bias입니다. $\sigma_{shape}=0.2$에서 compact condition을 equivariant로 둘 때 compact/extended mean $\Delta$가 대략 $0.19/0.07$, assignment를 바꾸면 $0.07/0.19$로 뒤집힙니다.

즉 method가 “두 mode를 완전히 동등하게 조합한다”기보다 **reference-frame condition이 더 강한 controller로 남아 있습니다.**

---

## 11. Result 2: ensemble property control은 calibrated regression이 아니다

Pocket + pharmacophore condition을 유지하면서 PSA 또는 mean pairwise RMSD target을 추가합니다.

PSA target은 80, 100, 120, 140 Å²를 sweep하고, RMSD는 1.0, 1.5, 2.0, 2.5 Å를 sweep합니다.

결과는 target에 따라 monotonic하게 움직입니다.

- mean PSA: target sweep에 따라 약 111 → 163 Å²
- mean pairwise RMSD: 약 1.69 → 2.17 Å

하지만 target value를 정확히 맞추는 calibrated conditional regressor처럼 동작하지는 않습니다. Extreme target에서는 baseline 쪽으로 압축됩니다.

논문의 올바른 해석은:

> **property target 방향으로 ensemble characteristic을 steering할 수 있다.**

이지,

> **요청한 ensemble PSA/RMSD 값을 정밀하게 생성한다.**

가 아닙니다.

Pocket condition도 완전히 무너지지는 않습니다. 저자 보고 interaction recovery는 모든 property run에서 약 0.944–0.961 범위로 baseline 0.954와 비슷하게 유지됩니다. 다만 이 interaction recovery는 **condition으로 제공한 reference interaction을 2 Å 이내에서 재현했는지** 보는 fidelity metric이지 새로운 favorable interaction이나 실제 affinity를 측정하는 지표는 아닙니다.

---

## 12. Result 3: dual-target AChE / MAO-B

Case study는 두 reference ligand의 pharmacophore mode를 합칩니다.

- AChE: donepezil, PDB 4EY7
- MAO-B: safinamide, PDB 2V5Z
- joint composition: $\gamma=2$, $\alpha=(0.5,0.5)$
- 100 generated molecules
- evaluation: AutoDock Vina, exhaustiveness 32

두 single-reference control과 비교했을 때, 저자들은 두 reference docking score에서 각각 1.0 kcal/mol 이내에 동시에 드는 fraction을 다음처럼 보고합니다.

| Generation condition | Both targets within 1.0 kcal/mol |
| --- | ---: |
| joint AChE + MAO-B | 23% |
| donepezil-only | 9% |
| safinamide-only | 7% |

0.5 kcal/mol threshold에서는 10% vs 1% / 3%, reference score 자체를 threshold로 잡으면 4% vs 1% / 0%입니다.

Best joint candidate의 Vina score는 저자 보고 기준

- AChE: $-12.24$ vs donepezil $-12.19$ kcal/mol
- MAO-B: $-11.85$ vs safinamide $-10.31$ kcal/mol

입니다.

이 결과가 지지하는 것은 **joint condition이 single condition보다 dual-oracle hit rate를 올릴 수 있다**는 것입니다.

지지하지 않는 것은:

- 실제 biochemical dual inhibition,
- selectivity panel,
- permeability/ADME,
- synthetic accessibility의 prospective success,
- docking-score difference가 실제 $\Delta G$ difference라는 주장

입니다.

또 100-sample case study이므로 생성 success rate의 uncertainty도 함께 생각해야 합니다. 이 숫자를 foundation-model-level general benchmark처럼 읽는 것은 과합니다.

---

## 13. Result 4: active A2A state를 target하고 inactive state를 피하기

두 번째 case study는 negative design을 더 직접적으로 보여줍니다.

- active A2A receptor: PDB 5G53, NECA-bound
- inactive receptor: PDB 4EIY, ZM241385-bound

Inactive pocket을 active frame에 align한 뒤, active/inactive에서 residue center-of-mass shift가 큰 residue만 남겨 state-discriminating pocket signal을 만듭니다.

두 threshold는

- 1.5 Å → 6 residues
- 2.0 Å → 2 residues

를 유지합니다.

Generation은

$$
\gamma=2,
\qquad
\alpha=(2,-1)
$$

을 사용합니다. 즉 active condition을 강하게 target하면서 inactive condition은 negative weight로 밀어냅니다.

Evaluation metric은

$$
\Delta\mathrm{Vina}_{state}
=
\mathrm{Vina}_{inactive}
-
\mathrm{Vina}_{active}.
$$

Vina는 낮을수록 좋으므로 양수면 active state가 상대적으로 더 favorable합니다.

저자 보고 mean은:

| Condition | mean $\Delta\mathrm{Vina}_{state}$ |
| --- | ---: |
| active-only | +0.47 kcal/mol |
| active + inactive-negative, 6 residues | +0.97 kcal/mol |
| active + inactive-negative, 2 residues | +1.26 kcal/mol |
| NECA reference | +1.61 kcal/mol |
| ZM241385 reference | +0.17 kcal/mol |

그리고 generated molecules의 NECA ECFP similarity median은 0.12–0.19 수준이라고 보고합니다.

이 case가 흥미로운 이유는 post-hoc counter-screening이 아니라 **inactive-state 정보가 sampling vector field에 직접 들어간다**는 점입니다.

하지만 가장 중요한 caveat도 같습니다.

$$
\Delta\mathrm{Vina}_{state}>0
\not\Rightarrow
\text{agonism}.
$$

Agonist efficacy는 receptor conformational thermodynamics, signaling pathway, kinetics를 포함합니다. 이 실험은 어디까지나 **state-specific docking proxy를 generation control로 사용할 수 있는지**를 보여줍니다.

---

## 14. 이 논문의 진짜 novelty와 덜 중요한 부분

### 가장 중요한 novelty 1 — training-time joint labels 없이 inference-time composition

Multi-state pair를 명시적으로 학습하지 않고 single-condition model call을 합칩니다.

이것은 새로운 target/off-target pair가 나타날 때마다 data assembly와 retraining을 요구하지 않는다는 점에서 실용적입니다.

### 가장 중요한 novelty 2 — negative condition이 first-class control

많은 conditional generator가 “무엇을 원하는가”는 표현하지만 “무엇을 피해야 하는가”는 post-hoc filter에 맡깁니다.

여기서는

$$
\alpha_k<0
$$

을 직접 control API로 둡니다. SBDD의 selectivity/active-inactive state 문제에 자연스럽습니다.

### 중요한 novelty 3 — frame conflict를 explicit design problem으로 취급

서로 다른 구조 condition은 coordinate frame 문제를 반드시 가집니다. 이를 무시하고 equivariant embedding을 단순 합산하지 않고, one-frame-equivariant / others-invariant contract를 둔 것은 재사용 가치가 큽니다.

### 덜 중요한 novelty — ensemble이라는 이름 자체

현재 paper에서 실제 ensemble property는 PSA와 pairwise RMSD 두 개이고, evaluation ensemble도 approximate합니다. 따라서 “Boltzmann-aware molecular design을 해결했다”보다 **multi-condition composable generator + preliminary ensemble-property control**로 읽는 편이 정확합니다.

---

## 15. Reproducibility: 공개 코드는 어느 수준인가

Official repository는 research code이지만 단순 teaser 수준은 아닙니다.

Repository snapshot `5d557afd...`에서 확인되는 공개 surface는 다음과 같습니다.

- training code
- unconditional sampling
- shape / pharmacophore sampling
- pocket-conditioned evaluation
- multi-condition evaluation
- symmetry evaluation
- preprocessing / split preparation
- tests
- evaluation documentation

README는 Zenodo record에서 다음 artifacts를 제공한다고 명시합니다.

- `enscond.ckpt`
- `geomdrugs-enscond.tar.gz`
- `spindr-enscond.tar.gz`
- `multi-cond-test.tar.gz`

즉 checkpoint만 공개한 것이 아니라 processed split/benchmark artifact까지 연결되어 있어 **paper-level rerun contract는 비교적 좋습니다.**

Evaluation docs도 reproducibility caveat를 명시합니다.

- checkpoint
- split/pair manifest
- seed
- integrator settings
- guidance weights
- docking/scoring parameters

를 함께 기록해야 하고, 새 split을 다시 생성하는 것만으로는 원래 benchmark를 정확히 재현할 수 없다고 적습니다.

따라서 Reproducibility를 High로 두는 것은 합리적이지만, “완전히 environment-independent한 exact reproduction”을 의미하지는 않습니다. Docking/xTB/RDKit/MMFF stack은 library/device/version에 민감하고 stochastic sampling도 포함됩니다.

---

## 16. 실패 가능성과 confounder

### 16.1 Reference-frame owner bias

Invariant condition이 equivariant condition보다 약합니다. Multi-condition weight만 같게 둔다고 실제 influence가 같아지지 않습니다.

따라서 condition weight $\alpha$와 frame ownership을 분리해서 ablation해야 합니다.

### 16.2 Inference cost는 condition 수에 선형 증가

Sampling step마다 null prediction과 각 $K$ condition prediction을 계산합니다.

대략 generator call cost는

$$
C_{sample}
\propto
T(K+1)
$$

입니다.

같은 frame에 있는 condition을 bundle할 수 있지만, independent reference frame이 많아지면 그대로 비싸집니다.

### 16.3 Two-mode evidence를 arbitrary-$K$ evidence로 읽으면 안 된다

Formulation은 arbitrary $K$를 허용하지만 main multi-mode benchmark는 두 mode로 제한됩니다. 저자들도 larger simultaneous reference-frame validation을 future work로 둡니다.

### 16.4 Evaluation oracle coupling

Dual-target와 active/inactive case 모두 Vina가 핵심 oracle입니다. Generator가 직접 Vina를 differentiable reward로 최적화한 것은 아니지만, method utility를 판단하는 final endpoint가 docking score에 크게 의존합니다.

따라서 실제 SBDD 적용에서는 반드시 independent higher-fidelity evaluator가 필요합니다.

### 16.5 Ensemble approximation mismatch

Training condition과 benchmark computation이 CREST vs ETKDG/MMFF로 다릅니다. 특히 flexibility-like property에서 ranking distortion 가능성이 있습니다.

### 16.6 Flexible-size model도 size bias가 남는다

Best ablation에서도 generated mean size가 training mean보다 약 2.6 heavy atoms 작습니다. Size matching을 하지 않은 metric에서는 이 bias가 property, docking, novelty를 모두 움직일 수 있습니다.

---

## 17. SBDD에서 바로 가져오려면 어떤 실험이 가장 유용한가

이 논문을 그대로 재현하는 것보다 **target/avoid condition이 실제 independent ranking을 개선하는지**를 먼저 검증하는 것이 정보량이 큽니다.

### Experiment A — target + close off-target negative generation

같은 receptor family 안에서 구조가 유사한 target/off-target pair를 고릅니다.

비교:

1. target-only generation
2. target-only generation + post-hoc off-target counter-screening
3. target + negative off-target compositional generation

고정:

- base checkpoint
- sample count
- total compute budget 또는 generator-call budget
- molecule-size distribution control
- docking preparation protocol

Primary evaluation은 generator가 사용한 control과 독립적이어야 합니다.

예를 들면:

- primary: held-out docking/rescoring engine 또는 physics-based interaction score
- diagnostic: original guidance-aligned Vina
- geometry: PoseBusters, clash, strain, pocket occupancy
- chemistry: validity, uniqueness, scaffold diversity, MW/QED/rotatable bonds

핵심 질문은

$$
\text{negative generation}
>
\text{generate then filter}
?
$$

입니다.

같은 compute budget에서 이 비교가 이겨야 generation-time negative design의 실질적 가치가 생깁니다.

### Experiment B — frame-owner swap

동일한 two-pocket condition에 대해

- target equivariant / off-target invariant
- target invariant / off-target equivariant
- pre-aligned both equivariant

를 비교합니다.

Weight $\alpha$를 고정하고 influence imbalance를 측정하면 adaptive symmetry가 실제로 condition semantics와 독립적인지 볼 수 있습니다.

### Experiment C — condition count scaling

$K=1,2,3,4$로 늘리면서

- success rate,
- validity,
- diversity,
- per-condition satisfaction,
- wall time,
- generator calls

을 같이 측정합니다.

Paper의 수식은 arbitrary $K$를 허용하지만 evidence는 주로 $K=2$입니다. 이 gap을 직접 찌르는 실험입니다.

### Experiment D — docking proxy independence

Generation/control은 Vina를 직접 쓰지 않더라도 최종 case-study metric이 Vina입니다. 따라서 generated set을 고정한 뒤

- Vina,
- 다른 docking/scoring model,
- geometry-only pose quality,
- 가능하면 short relaxation/interaction energy

사이 rank correlation과 top-set overlap을 봅니다.

효과가 Vina에서만 보인다면 “multi-state design”보다 evaluator-specific success일 가능성이 큽니다.

---

## 18. 다른 generative-control 전략과의 위치

이 논문은 conditional generation, post-training, optimization을 다음처럼 구분해서 보면 좋습니다.

| Strategy | New objective가 생기면 | Sampling cost | Distribution 변화 | 주요 위험 |
| --- | --- | --- | --- | --- |
| joint-condition retraining | 다시 학습 | 보통 | checkpoint 자체 변경 | matched data 필요 |
| [[papers/generative-models/flow-matching-rl|reward-weighted post-training]] | post-train | 낮음 | checkpoint 자체 변경 | reward hacking / prior drift |
| post-hoc generate-and-filter | 재학습 없음 | 후보 많이 필요 | generator unchanged | 낮은 hit rate / oracle cost |
| ensemble-conditioned composition | 재학습 없음 | $K$에 따라 증가 | sampling field만 변경 | field conflict / frame bias |

따라서 이 방법이 가장 매력적인 상황은:

- condition 조합이 자주 바뀌고,
- 각 condition에 single-condition model behavior가 이미 학습되어 있으며,
- sample-time 추가 forward pass를 감당할 수 있고,
- matched multi-state dataset은 부족한 경우

입니다.

반대로 objective가 하나로 고정되고 대규모 sampling을 반복한다면, 충분한 data가 있을 경우 dedicated fine-tuning/post-training이 더 효율적일 수도 있습니다.

---

## 19. 최종 판단

이 논문은 **새로운 3D generator backbone**보다 **condition composition layer** 때문에 저장할 가치가 큽니다.

특히 SBDD에서 반복되는

$$
\text{target}
+
\text{counter-target}
+
\text{conformational state}
$$

문제를 하나의 generator API로 표현할 수 있다는 점이 강합니다.

가장 설득력 있는 evidence는 두 가지입니다.

1. controlled two-shape benchmark에서 target/avoid behavior가 directionally 나타납니다.
2. AChE/MAO-B와 active/inactive A2A 사례에서 추가 condition이 single-condition control보다 원하는 Vina-based objective를 개선합니다.

하지만 논문이 아직 증명하지 않은 것도 명확합니다.

- arbitrary many-state composition의 안정성,
- frame-owner bias가 사라지는지,
- higher-fidelity physics에서도 advantage가 유지되는지,
- experimental selectivity/agonism,
- thermal ensemble property를 정확히 제어하는지

입니다.

따라서 가장 안전한 verdict는:

> **Composable negative design을 3D molecular Flow Matching에 실용적인 형태로 넣은 강한 architecture/control idea다. 다만 현재 evidence는 two-mode + docking/approximate-ensemble proxy에 집중되어 있으므로, 실제 SBDD에서는 independent rescoring과 matched compute-budget generate-vs-filter control을 통과해야 한다.**

---

## Three durable takeaways

1. **Multi-state design은 paired multi-state training data가 없어도 가능할 수 있다.** Single-condition endpoint/vector fields를 inference에서 composition하는 방식이 하나의 실용적인 해법입니다.

2. **Negative condition은 post-hoc filter가 아니라 generation dynamics의 first-class control이 될 수 있다.** Target/off-target, active/inactive state 문제에 특히 자연스럽습니다.

3. **여러 3D condition을 합칠 때 진짜 어려운 문제는 weight보다 coordinate frame입니다.** Adaptive symmetry는 흥미로운 해법이지만 reference-frame owner bias가 남으므로, 실제 적용에서는 frame assignment 자체를 ablation해야 합니다.

---

## Related

- [[papers/generative-models/flow-matching-rl|Controllable Molecular Generation with Fine-Tuned Flow Matching]]
- [[papers/generative-models/lift|LiFT — Language-Informed Flow Matching]]
- [[papers/sbdd/surfspec|SurfSpec]]
- [[concepts/generative-models/flow-matching|Flow Matching]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]

## Sources

- [Paper — arXiv:2609.15077](https://arxiv.org/abs/2609.15077)
- [Official arXiv HTML with figures](https://arxiv.org/html/2609.15077v1)
- [Official code — rssrwn/ensemble-cond-design](https://github.com/rssrwn/ensemble-cond-design)
- [Released checkpoint and processed datasets — Zenodo 22485204](https://zenodo.org/records/22485204)
