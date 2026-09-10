---
title: BackFlip-2 — Predicting directional flexibility in proteins
aliases:
  - papers/backflip-2
  - papers/protein-modeling/backflip-2
tags:
  - papers
  - protein-modeling
  - protein-dynamics
  - geometric-deep-learning
  - equivariance
  - molecular-dynamics
  - flexibility
status: full-note
source_type: ArXiv
source_url: https://arxiv.org/abs/2609.08474
---

# BackFlip-2: Predicting directional flexibility in proteins

> **한 줄 요약:** BackFlip-2는 equilibrium protein structure 하나에서 residue별 **3×3 anisotropic covariance tensor**와 residue-pair **dynamic coupling / DCCM**을 직접 예측해, scalar RMSF가 버리는 *어느 방향으로 움직이는가*와 *어떤 residue들이 함께 움직이는가*를 매우 저렴한 구조 annotation으로 복원하려는 SE(3)-equivariant surrogate입니다.

## 왜 이 논문을 저장하는가

Protein structure model이나 docking pipeline은 대개 protein을 한 장의 정적 구조로 봅니다. 하지만 실제 binding site에서 중요한 것은 단순히 residue가 얼마나 흔들리는지가 아니라,

- 어느 방향으로 흔들리는가,
- neighboring residue가 같은 방향으로 움직이는가,
- 서로 떨어진 domain이 correlated / anticorrelated motion을 보이는가,
- apo 구조의 motion direction이 functional conformational change와 정렬되는가

입니다.

가장 흔한 cheap descriptor인 RMSF는 이 정보를 거의 모두 압축합니다.

$$
\mathrm{RMSF}_i = \sqrt{\operatorname{tr}(\Sigma_i)}.
$$

여기서 $\Sigma_i\in\mathbb{R}^{3\times3}$는 residue $i$의 fluctuation covariance입니다. Trace만 남기면 세 축의 variance와 principal direction이 사라집니다.

BackFlip-2가 흥미로운 이유는 이 covariance 자체를 **equivariant target**으로 만들었다는 데 있습니다. Protein을 회전시키면 predicted covariance도 같은 방식으로 회전해야 합니다.

$$
\hat\Sigma_i(RX+t)=R\hat\Sigma_i(X)R^\top.
$$

동시에 residue-pair dynamics는 invariant scalar로 예측합니다. 즉 논문은 protein flexibility를 하나의 scalar regression으로 보지 않고,

$$
\boxed{
\text{static structure}
\rightarrow
\text{directional local motion}
+
\text{long-range dynamic coupling}
}
$$

이라는 **representation problem**으로 다시 정의합니다.

이 관점은 SBDD에서도 재사용 가치가 있습니다. BackFlip-2 자체가 docking/affinity model은 아니지만, static pocket representation에 cheap dynamics descriptor를 붙일 수 있는 후보이기 때문입니다.

---

## Metadata

| Field | Value |
| --- | --- |
| Paper | Predicting directional flexibility in proteins |
| Authors | Vsevolod Viliuga, Leif Seute, Matteo Tadiello, Nicolas Wolf, Frauke Gräter, Arne Elofsson |
| arXiv | 2609.08474 |
| Submitted | 2026-09-08 |
| Model | BackFlip-2 |
| Input | equilibrium protein backbone frames + amino-acid identity |
| Main outputs | per-residue covariance, pairwise coupling, DCCM |
| Main MD sources | ATLAS, mdCATH |
| Backbone | modified AlphaFold2 Invariant Point Attention encoder |
| Parameters | 965K reported for BackFlip-2 |
| Training | 200 epochs; authors report ~4 GPU-hours on one A100 |
| Official code | https://github.com/graeter-group/backflip |
| Code snapshot inspected | `887c6a73b216e3fe2ff9f551ae6e2c983c21b292` |
| Released checkpoints | ATLAS, mdCATH, joint ATLAS+mdCATH variants |
| Paper license | CC BY 4.0 |
| Repository software license | MIT |

> **Claim boundary:** 아래 수치와 biological examples는 저자들이 paper/repository에서 보고한 결과입니다. MD trajectory와의 agreement는 experimental function, ligand binding, affinity, induced fit 또는 long-timescale conformational transition을 자동으로 증명하지 않습니다.

---

## Figure guide — 이 두 그림이 핵심 intuition을 잡아준다

### Figure A — scalar RMSF보다 covariance가 필요한 이유

![BackFlip-2 equivariant covariance ellipsoids](https://raw.githubusercontent.com/graeter-group/backflip/887c6a73b216e3fe2ff9f551ae6e2c983c21b292/assets/exp_ellipsoids_new.png)

*Official BackFlip repository asset corresponding to the paper's directional-covariance comparison. Source: [BackFlip repository asset](https://github.com/graeter-group/backflip/blob/887c6a73b216e3fe2ff9f551ae6e2c983c21b292/assets/exp_ellipsoids_new.png), paper: [arXiv:2609.08474](https://arxiv.org/abs/2609.08474). The project repository is MIT-licensed and the paper is CC BY 4.0. No scientific content was modified.*

**볼 것:** MD에서 관찰되는 residue fluctuation은 대체로 spherical하지 않습니다. Equivariant model은 길쭉한 ellipsoid의 orientation까지 따라가지만, non-equivariant head는 방향을 안정적으로 정의할 수 없어 훨씬 isotropic한 prediction으로 수렴합니다.

이 그림이 지지하는 것은 `equivariance가 멋있다`는 일반론이 아닙니다. **target 자체가 coordinate-frame-dependent tensor라면 output transformation law를 architecture에 넣는 것이 정보 보존에 직접 필요하다**는 주장입니다.

### Figure B — 한 구조에서 RMSF와 long-range coupling을 동시에 얻는다

![BackFlip-2 ubiquitin RMSF and DCCM prediction](https://raw.githubusercontent.com/graeter-group/backflip/887c6a73b216e3fe2ff9f551ae6e2c983c21b292/assets/1ubq_backflip_flexibility_prediction.png)

*Official BackFlip repository inference example for ubiquitin (1UBQ). Source: [BackFlip repository asset](https://github.com/graeter-group/backflip/blob/887c6a73b216e3fe2ff9f551ae6e2c983c21b292/assets/1ubq_backflip_flexibility_prediction.png). The upper profile is RMSF derived from the predicted covariance; the matrix is the predicted DCCM. Author-produced visualization, not independent validation.*

**볼 것:** RMSF는 diagonal/local flexibility만 보여주지만 DCCM은 residue pair의 correlated and anticorrelated motion을 보여줍니다. BackFlip-2는 이 둘을 별도의 expensive trajectory generation 없이 동일한 structure-conditioned encoder에서 예측합니다.

논문의 adenylate kinase functional example도 중요합니다. Apo structure에서 LID/NMP-binding domain의 covariance direction과 negative inter-domain coupling이 open→closed motion과 관련된 축을 포착한다고 저자들은 해석합니다. 해당 비교는 paper Figure 3에서 보는 것이 가장 정확합니다: [paper PDF](https://arxiv.org/pdf/2609.08474).

---

## 1. Problem: flexibility를 scalar 하나로 줄이면 무엇을 잃는가

MD trajectory에서 residue $i$의 mean-centered displacement를

$$
\delta x_i(t)=x_i(t)-\langle x_i\rangle
$$

라고 하면 가장 기본적인 second moment는

$$
\Sigma_i
=
\left\langle
\delta x_i\delta x_i^\top
\right\rangle
\in\mathbb{R}^{3\times3}
$$

입니다.

이 matrix는 세 가지를 동시에 포함합니다.

1. total amplitude,
2. anisotropy,
3. principal motion direction.

RMSF는

$$
\mathrm{RMSF}_i^2=\operatorname{tr}(\Sigma_i)
$$

이므로 total amplitude만 남깁니다. 두 residue가 RMSF는 같아도 하나는 hinge 방향으로 길게 움직이고 다른 하나는 isotropic하게 흔들릴 수 있습니다.

Protein function이나 pocket adaptation에서는 이 차이가 중요할 수 있습니다. 특히 ligand approach direction, loop opening, domain closure처럼 특정 direction이 필요한 경우 scalar flexibility만으로는 geometry를 충분히 표현하지 못합니다.

BackFlip-2의 첫 번째 연구 질문은 따라서 단순합니다.

> **정적 구조 하나에서 residue별 covariance tensor를 직접 예측할 수 있는가?**

두 번째 질문은 더 어렵습니다.

> **trajectory 없이 residue-residue dynamic coupling까지 복원할 수 있는가?**

---

## 2. Output contract: local tensor와 pairwise scalar를 분리한다

### 2.1 Per-residue covariance

Residue $i$에 대해 target은

$$
\Sigma_i\in \mathbb{S}_+^3
$$

입니다. 즉 symmetric positive-semidefinite (SPSD) matrix입니다.

좌표를 $R\in SO(3)$로 회전하면

$$
\Sigma_i' = R\Sigma_iR^\top.
$$

이것은 invariant scalar target이 아닙니다. Tensor target의 orientation도 prediction의 일부입니다.

### 2.2 Pairwise coupling

Full cross-covariance $\Sigma^{(ij)}$를 그대로 예측하는 대신 paper는 pairwise scalar coupling을

$$
C_{ij}=\operatorname{tr}(\Sigma^{(ij)})
$$

형태로 다룹니다.

그리고 normalized form인 DCCM은

$$
\widetilde C_{ij}
=
\frac{C_{ij}}
{\mathrm{RMSF}_i\mathrm{RMSF}_j}
$$

로 계산합니다.

이 값은 대략 $[-1,1]$ 범위에서

- positive: correlated motion,
- negative: anticorrelated motion,
- near zero: weak linear coupling

을 나타냅니다.

여기서 설계가 깔끔합니다.

- direction이 본질적인 per-residue motion → equivariant matrix,
- pair relation의 signed strength → invariant scalar.

**Target의 transformation law에 맞춰 output type을 분리한 것**이 architecture의 핵심입니다.

---

## 3. Input representation: protein backbone을 residue frames로 본다

BackFlip-2는 protein backbone을 residue-wise rigid frame으로 표현합니다.

$$
T_i=(x_i,R_i),
$$

where

- $x_i$ is a residue position,
- $R_i$ is the residue-local orientation.

Encoder는 modified AlphaFold2 Invariant Point Attention(IPA)을 사용합니다. Paper가 기술하는 main setting은 대략

- node dimension 96,
- pair/edge dimension 64,
- 4 IPA layers,
- backbone frame,
- one-hot amino-acid type,
- positional encoding,
- pair distance features up to 20 Å

입니다.

중요한 점은 backbone frames가 structure generation처럼 update되지 않는다는 것입니다. 여기서는 이미 주어진 equilibrium structure를 읽고 dynamics descriptor를 예측합니다.

따라서 task는

$$
\{T_i,a_i\}_{i=1}^N
\xrightarrow{f_\theta}
\{\hat\Sigma_i\}_{i=1}^N,
\{\hat C_{ij}\}_{i,j=1}^N
$$

입니다.

---

## 4. Covariance head: SPSD와 equivariance를 동시에 보장한다

이 paper에서 가장 재사용 가치가 높은 부분입니다.

Encoder가 residue-local invariant/equivariant representation을 만든 뒤 covariance head는 local frame에서 임의의 $3\times3$ matrix $A_i$를 예측합니다.

그 다음

$$
\Sigma_i^{\mathrm{local}}=A_iA_i^\top
$$

로 둡니다.

이 구성은 자동으로

$$
\Sigma_i^{\mathrm{local}}\succeq0
$$

을 만족합니다.

마지막으로 residue frame orientation을 이용해 global coordinate로 보냅니다.

$$
\hat\Sigma_i
=
R_i\Sigma_i^{\mathrm{local}}R_i^\top.
$$

Global protein rotation $Q$를 적용하면 local covariance는 그대로이고 residue frame이 $QR_i$가 되므로

$$
\hat\Sigma_i'
=
(QR_i)\Sigma_i^{\mathrm{local}}(QR_i)^\top
=
Q\hat\Sigma_iQ^\top.
$$

즉 output equivariance가 구조적으로 보장됩니다.

이 방식은 `higher-order irrep network를 반드시 깊게 쌓아야 tensor를 예측할 수 있다`는 것과는 다릅니다. **Local invariant prediction + known frame transport**로 physically valid tensor output을 만들 수 있습니다.

---

## 5. Pairwise coupling head

Coupling head는 pair embedding을 읽어 scalar $C_{ij}$를 예측합니다. Pairwise coupling은 residue ordering을 바꿔도 대칭이어야 하므로 prediction을 symmetrize합니다.

$$
\hat C_{ij}=\hat C_{ji}.
$$

이후 predicted covariance에서 얻은 RMSF와 coupling으로 DCCM을 구성할 수 있습니다.

Representation 관점에서 특히 흥미로운 것은 **node tensor + dense pair scalar**의 조합입니다.

정적 protein model에 가져오면 다음과 같은 feature contract를 만들 수 있습니다.

```text
residue node:
  static geometry
  amino-acid identity
  covariance tensor / derived invariants

residue pair:
  distance/orientation
  predicted dynamic coupling
  predicted DCCM
```

이 구조는 protein-ligand pairformer류 모델에서 dynamics를 별도의 pair channel로 넣는 아이디어와 자연스럽게 연결됩니다.

---

## 6. Training objective

Paper는 covariance matrix 자체의 discrepancy, scalar amplitude, pair coupling을 하나로 묶어 학습합니다.

핵심 supervision은 대략 다음 세 층입니다.

1. covariance geometry,
2. RMSF / magnitude,
3. DCCM / coupling.

Covariance comparison에는 covariance-matrix distance(CMD) 계열을 사용하고, RMSF MAE와 DCCM MSE를 함께 최적화합니다.

이 multi-objective가 필요한 이유는 matrix loss 하나만으로는 downstream에서 읽기 쉬운 amplitude와 pair correlation 품질을 충분히 통제하지 못할 수 있기 때문입니다.

Paper의 main model은 200 epochs 학습되고, 저자들은 single A100에서 약 4 GPU-hours 수준의 training cost를 보고합니다. Checkpoint selection은 validation loss 기준입니다.

---

## 7. Data contract: 무엇을 ground truth dynamics로 보는가

### ATLAS

ATLAS setting은 paper의 주요 training/evaluation source입니다.

저자 설명 기준:

- 1,390 proteins,
- protein당 3 independent trajectories,
- trajectory당 100 ns,
- CHARMM36m,
- 300 K,
- 2 fs timestep.

Trajectory에서 residue covariance와 cross-residue coupling을 계산해 supervised labels로 사용합니다.

### mdCATH

mdCATH는 domain-level MD collection으로 cross-dataset transfer를 보는 데 사용됩니다.

Paper는

- 5,398 domains,
- multiple temperatures,
- 320 K subset을 주요 comparison에 사용,
- total 500 ns-scale trajectories,
- CHARMM22,
- 4 fs timestep

이라는 protocol 차이를 다룹니다.

이 차이는 단순 dataset shift가 아닙니다. Force field, temperature, timestep, domain composition, simulation protocol이 모두 달라질 수 있습니다.

따라서 BackFlip-2가 배우는 target을 더 정확하게 쓰면

> **protein의 절대적이고 유일한 dynamics**

가 아니라

> **특정 MD protocol과 sub-microsecond observation window에서 정의된 second-order fluctuation statistics**

입니다.

이 distinction은 downstream application에서 반드시 유지해야 합니다.

---

## 8. Evaluation: 하나의 metric으로 tensor prediction을 판단하면 안 된다

Paper는 flexibility를 여러 axis로 나눠 평가합니다.

| Metric | 무엇을 본다 |
| --- | --- |
| RMSF Pearson | residue-wise amplitude ranking |
| RMSF MAE | absolute amplitude error |
| RMWD | covariance matrix geometry discrepancy |
| symmetric KL | Gaussian covariance distribution mismatch |
| ellipsoid IoU / Dice | anisotropic shape + orientation overlap |
| DCCM Pearson | pair coupling pattern |
| DCCM MAE | pair coupling absolute error |

특히 ellipsoid overlap이 중요합니다. RMSF가 비슷한 두 covariance라도 principal axis가 다르면 directional dynamics prediction은 틀릴 수 있습니다.

---

## 9. Main result 1: cheap RMSF surrogate를 넘어선다

저자 보고 ATLAS FlexPert split에서 대표 결과는 다음과 같습니다.

| Method | RMSF Pearson ↑ | RMSF MAE ↓ | Params | Inference |
| --- | ---: | ---: | ---: | ---: |
| MD reference trajectory-vs-trajectory | 0.88 | 0.47 | — | $O(10^4)$ s-scale |
| FlexPert | 0.83 | 0.80 | 1.2B | ~0.4 s |
| Pegasus | 0.75 | 0.82 | 11M | ~2.5 s |
| BackFlip | 0.84 | 0.62 | 321K | ≤0.02 s |
| **BackFlip-2** | **0.87** | **0.58** | **965K** | **≤0.02 s** |

이 결과의 올바른 읽기는 `BackFlip-2가 MD를 대체했다`가 아닙니다.

MD trajectory 두 개 사이에도 finite-sampling discrepancy가 존재하고, BackFlip-2는 MD-derived descriptors를 빠르게 근사합니다. 즉 **trajectory-level microstates를 생성하는 대신 second-order summary statistics를 amortize**한 것입니다.

De novo protein subset에서도 저자들은 BackFlip-2가 기존 cheap predictors보다 더 높은 RMSF agreement를 보고합니다. 이 결과는 training-set memorization만으로 설명하기 어려운 signal을 주지만, de novo set 하나가 모든 structural OOD를 보장하는 것은 아닙니다.

---

## 10. Main result 2: ensemble model과 비교하면 trade-off가 보인다

AlphaFlow-style split에서 paper가 보고한 대표 값은 다음과 같습니다.

| Method | RMWD ↓ | Sym. KL ↓ | RMSF r ↑ | RMSF MAE ↓ | DCCM r ↑ | DCCM MAE ↓ | Inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BackFlip-2 | 0.88 | 0.67 | 0.89 | 0.35 | 0.80 | 0.16 | ≤0.02 s |
| AFMD-T | 0.84 | 0.60 | 0.93 | 0.25 | 0.89 | 0.12 | ~8,200 s |
| BBFlow | 0.93 | 0.65 | 0.93 | 0.42 | 0.86 | 0.16 | ~160 s |
| DynaProt | 1.18 | 0.91 | 0.87 | — | 0.66 | — | ~0.02 s |

BackFlip-2는 covariance quality에서 훨씬 비싼 ensemble methods와 경쟁력이 있지만, **DCCM coupling은 AFMD-T/BBFlow보다 약합니다.**

이 trade-off가 중요합니다.

- 필요한 것이 cheap annotation이라면 BackFlip-2의 속도 이점이 큽니다.
- alternative conformations나 long-range coupling accuracy가 핵심이면 ensemble generation이 여전히 더 많은 정보를 제공합니다.

즉 `fast dynamics surrogate`와 `full ensemble generator`는 같은 product가 아닙니다.

---

## 11. Equivariance ablation: 왜 방향성 target에는 symmetry가 실제로 필요한가

Paper의 directional ellipsoid overlap에서 저자 보고값은 대략 다음과 같습니다.

| Model | IoU ↑ | Dice ↑ |
| --- | ---: | ---: |
| BackFlip-2 equivariant | **0.49** | **0.64** |
| non-equivariant variant | 0.42 | 0.58 |
| BioEmu | 0.25 | 0.38 |
| AFMD-T | 0.48 | 0.63 |
| BBFlow | 0.47 | 0.62 |

Non-equivariant model은 RMSF amplitude는 어느 정도 맞출 수 있어도 covariance ellipsoid가 near-spherical해지는 경향을 보입니다.

왜 그런지 생각하면 자연스럽습니다. Global frame에서 arbitrary direction을 예측하게 하면 protein을 회전시켰을 때 prediction direction이 어떻게 바뀌어야 하는지 학습으로만 알아내야 합니다. Equivariant head는 그 constraint를 architecture에 고정합니다.

이 논문이 주는 일반 원칙은:

> **Invariant scalar benchmark만 보고 equivariant architecture의 가치를 판단하면 방향성 information gain을 놓칠 수 있다.**

입니다.

---

## 12. PSD parameterization ablation: 복잡한 Cholesky가 꼭 낫지는 않다

Appendix ablation은 covariance parameterization과 equivariance를 분리합니다.

대표적으로 저자들이 보고한 패턴은:

- equivariant + simple $AA^\top$: best/near-best overall,
- equivariant + Cholesky: 비슷하지만 명확한 우위 없음,
- non-equivariant variants: covariance geometry와 DCCM에서 더 큰 손실.

즉 improvement의 핵심은 `더 정교한 SPD parameterization`보다 **올바른 transformation law**에 있습니다.

이것은 implementation lesson으로 꽤 좋습니다. 물리 constraint를 넣을 때 가장 복잡한 parameterization부터 시작할 필요가 없습니다.

---

## 13. Cross-dataset transfer: mdCATH가 보여주는 것

ATLAS-trained model을 mdCATH 320 K subset에 그대로 적용하면 저자들은 대략

- RMWD 1.63,
- symmetric KL 2.57,
- RMSF Pearson 0.85,
- RMSF MAE 0.69,
- DCCM Pearson 0.81,
- DCCM MAE 0.15

를 보고합니다.

Joint ATLAS+mdCATH training은 mdCATH covariance metrics를 개선해 대략 RMWD 1.51, KL 1.72 수준으로 낮추지만 ATLAS에서 일부 metric은 약간 희생합니다.

이 결과는 중요한 경고를 줍니다.

$$
\text{dynamics label}
=
\text{protein structure}
+
\text{simulation protocol}
+
\text{timescale}
+
\text{force field}
+
\text{temperature}
+
\text{sampling noise}.
$$

따라서 dynamics foundation feature를 만들려면 dataset을 그냥 합치는 것보다 **condition/domain metadata를 명시적으로 모델링할 가능성**을 고려해야 합니다.

---

## 14. Input robustness: perfect crystal geometry만 요구하지 않는다

Paper appendix는 input coordinate perturbation과 AlphaFold-predicted structure를 사용한 robustness test를 제공합니다.

저자 보고상

- 0.2 Å noise,
- 0.5 Å noise,
- AlphaFold structure input

에서도 main metrics가 급격히 붕괴하지 않습니다. AlphaFold input의 median global RMSD가 수 Å 수준이어도 RMSF/covariance metrics가 상당 부분 유지됩니다.

이것은 실제 application에서 중요합니다. Dynamics annotation이 experimental structure에만 묶이면 large-scale proteome annotation에 쓰기 어렵기 때문입니다.

하지만 이것도 `predicted structure에서 pocket dynamics가 정확하다`는 뜻은 아닙니다. Global robustness와 local binding-site accuracy는 별도의 test가 필요합니다.

---

## 15. Functional examples: adenylate kinase를 어떻게 읽어야 하는가

Adenylate kinase는 open apo state와 closed holo state 사이에 큰 domain motion이 있는 classical example입니다.

저자들은 apo structure에서

- LID / NMP-binding regions가 상대적으로 flexible하고,
- covariance ellipsoid의 principal directions가 closure/opening axis와 관련되며,
- domains 사이에 anticorrelated coupling pattern이 나타나는 것

을 보여줍니다.

이것은 BackFlip-2가 단순 B-factor-like amplitude만 배우는 것이 아니라 **functionally interpretable direction/coupling signal**을 포착할 수 있다는 qualitative evidence입니다.

하지만 다음까지는 증명하지 않습니다.

$$
\text{predicted covariance direction}
\Rightarrow
\text{actual transition pathway probability}.
$$

Second moment는 transition path나 multimodal free-energy landscape 전체가 아닙니다.

---

## 16. 가장 중요한 limitation: Gaussian second moment와 sub-microsecond timescale

Covariance는 본질적으로 분포를 second-order moment로 압축합니다.

한 basin 안의 near-Gaussian fluctuation에는 좋은 descriptor일 수 있지만,

```text
state A  ←→  high barrier  ←→  state B
```

처럼 distinct metastable state가 있는 경우 하나의 covariance matrix는 두 state의 구조와 barrier를 표현하지 못합니다.

Paper 자체도 descriptor가 주로 300–500 ns 이하 MD regime에 맞춰져 있으며 millisecond-scale transition이나 strongly multimodal dynamics에는 ensemble/sampling model이 더 적합하다는 boundary를 둡니다.

따라서 BackFlip-2를 다음과 혼동하면 안 됩니다.

- MD simulator,
- conformational ensemble generator,
- free-energy estimator,
- transition-path sampler.

BackFlip-2는 **fast dynamics descriptor predictor**입니다.

---

## 17. SBDD에서 어디에 써볼 수 있는가

가장 자연스러운 사용은 docking model을 BackFlip-2로 교체하는 것이 아니라 protein representation에 side-channel을 추가하는 것입니다.

### 17.1 Residue node feature

Covariance에서 invariant descriptor를 뽑을 수 있습니다.

$$
\operatorname{tr}(\Sigma_i),
\quad
\log\det(\Sigma_i+\epsilon I),
\quad
\lambda_1,\lambda_2,\lambda_3,
\quad
\frac{\lambda_{\max}}{\sum_k\lambda_k}.
$$

이들은 flexibility magnitude와 anisotropy를 scalar node feature로 제공합니다.

### 17.2 Equivariant directional feature

Principal eigenvector를 쓰고 싶다면 global Cartesian vector를 scalar MLP에 그대로 넣으면 안 됩니다. Downstream model이 equivariant하다면

$$
(v_i,\lambda_i)
$$

를 proper vector/tensor channel로 전달하거나 residue-local frame으로 표현해야 합니다.

즉 BackFlip-2 feature를 사용할 때도 **coordinate contract**를 유지해야 합니다.

### 17.3 Pair feature

DCCM은 pairformer-style representation과 특히 잘 맞습니다.

$$
p_{ij}^{(0)}
\leftarrow
[p_{ij}^{\mathrm{static}},\hat C_{ij},\hat{\widetilde C}_{ij}].
$$

Static distance/contact와 별개로 `tend to move together / against each other`를 pair channel에 추가할 수 있습니다.

### 17.4 Pocket mobility prior

Pocket residue의 covariance principal axis가 cavity opening direction과 관련된다면 ligand placement/refinement에서 useful prior가 될 가능성이 있습니다.

하지만 이것은 아직 hypothesis입니다. Paper에는 protein-ligand pose/affinity experiment가 없습니다.

---

## 18. 내가 먼저 돌릴 matched ablation

BackFlip-2의 SBDD utility를 확인하려면 복잡하게 시작할 필요가 없습니다.

### Arms

| Arm | Protein dynamics feature |
| --- | --- |
| A | static structure only |
| B | + predicted scalar RMSF |
| C | + covariance eigenvalues / anisotropy invariants |
| D | + full equivariant covariance representation |
| E | + DCCM pair channel |
| F | + covariance + DCCM |

### Evaluation

동일 model capacity와 training budget에서 최소한 다음을 분리해 봅니다.

- pose top-1 / top-k,
- pocket-conditioned ranking or scoring,
- apo→holo robustness,
- protein-family OOD,
- pocket-similarity OOD,
- flexible-loop subset,
- rigid-pocket subset,
- inference overhead,
- cached-feature storage cost.

### 가장 중요한 control

MD-derived true descriptors를 teacher/oracle로 둔 upper-bound arm을 추가해야 합니다.

```text
static
vs predicted BackFlip-2 dynamics
vs MD-derived dynamics
```

Predicted dynamics가 도움이 안 된다면 두 가능성을 나눌 수 있습니다.

1. dynamics feature 자체가 task에 불필요하다.
2. useful하지만 predictor error가 information을 지운다.

이 구분 없이 `BackFlip-2 feature failed`라고 결론 내리면 안 됩니다.

---

## 19. 더 강한 test: apo/holo와 pocket-specific directionality

특히 SBDD에서는 global protein metric보다 pocket-local test가 더 중요합니다.

다음 protocol이 더 직접적입니다.

1. apo structure만 input으로 사용.
2. BackFlip-2 covariance를 예측.
3. holo ligand가 들어오면서 실제로 이동한 pocket residues의 displacement를 계산.
4. predicted principal covariance axis와 apo→holo displacement angle을 비교.

예를 들어

$$
\cos\theta_i
=
\frac{|v_i^\top\Delta x_i|}
{\|v_i\|\|\Delta x_i\|}
$$

를 볼 수 있습니다.

여기서 $v_i$는 predicted dominant fluctuation axis입니다.

이 metric이 random-axis / RMSF-only baseline보다 높다면 `directional flexibility`가 실제 pocket adaptation direction에 information을 준다는 더 직접적인 evidence가 됩니다.

주의할 점은 holo structure나 ligand pose에서 derived된 정보를 input preprocessing에 섞지 않는 것입니다. 그렇지 않으면 coordinate/template leakage가 생깁니다.

---

## 20. Reproducibility path

공식 repository는 비교적 좋은 reproduction surface를 제공합니다.

### Released artifacts

- BackFlip-2.1 ATLAS checkpoint,
- mdCATH checkpoint,
- joint ATLAS+mdCATH checkpoint,
- dataset preparation files,
- modified ATLAS features,
- inference API / CLI,
- covariance/DCCM evaluation script,
- explicit equivariance test script,
- Colab tutorial.

공식 Python API는 한 PDB에서

- `per_res_covariance` $(N,3,3)$,
- `pairwise_couplings` $(N,N)$,
- `pairwise_DCCM` $(N,N)$

을 반환합니다.

Repository는 batched A100 inference에서 약 50 proteins/s까지 가능하다고 기술하지만, 이 throughput은 hardware, length distribution, batching에 의존하는 **repository-reported figure**로 봐야 합니다.

### Dataset license caveat

Software는 MIT지만 modified ATLAS dataset은 upstream 조건에 따라 CC BY-NC 4.0으로 안내됩니다. 따라서 code license와 dataset license를 분리해 봐야 합니다.

---

## 21. What the paper establishes

이 paper의 evidence가 비교적 강하게 지지하는 것은 다음입니다.

1. Static equilibrium structure에서 MD-derived RMSF를 상당히 정확하게 amortize할 수 있습니다.
2. Scalar RMSF를 넘어 anisotropic per-residue covariance를 직접 예측할 수 있습니다.
3. Equivariant output construction이 directional covariance quality를 개선합니다.
4. Pairwise DCCM pattern도 cheap하게 예측할 수 있습니다.
5. ATLAS→mdCATH transfer와 coordinate perturbation/AF input robustness에서 일정 수준의 generalization signal이 있습니다.
6. Full ensemble methods보다 정보는 제한적이지만 inference cost가 극도로 낮습니다.

---

## 22. What the paper does NOT establish

다음 claim은 아직 별도 검증이 필요합니다.

### Ligand binding utility

Protein dynamics descriptor accuracy가 좋아도

$$
\text{better covariance}
\not\Rightarrow
\text{better docking / affinity / screening}.
$$

Paper에는 이 downstream evidence가 없습니다.

### Long-timescale transitions

Single covariance와 DCCM은 multimodal landscape와 rare transition kinetics를 복원하지 않습니다.

### Experimental dynamics equivalence

Ground truth의 중심은 MD-derived observable입니다. NMR/order parameters/HDX/B-factor와의 관계는 별도 measurement model이 필요합니다.

### Universal dynamics across simulation protocols

ATLAS와 mdCATH 차이 자체가 보여주듯 force field, temperature, simulation length가 target distribution에 영향을 줄 수 있습니다.

### Pocket-local OOD

Global protein metric이 유지되어도 flexible loop, cryptic pocket, membrane protein, intrinsically disordered region 같은 difficult regime에서 동일하다고 말할 수 없습니다.

---

## 23. Failure modes

| Failure mode | 왜 위험한가 |
| --- | --- |
| RMSF 성능만 보고 directional model을 평가 | anisotropy와 orientation information을 못 봄 |
| DCCM을 causal allostery로 해석 | correlation은 causation/energy pathway가 아님 |
| covariance를 conformational ensemble로 해석 | second moment는 multimodal states를 복원하지 못함 |
| one MD protocol을 biological truth로 취급 | force field/timescale/temperature dependence 무시 |
| principal vector를 scalar network에 그대로 넣음 | rotation-frame leakage / broken transformation contract |
| holo ligand-derived alignment을 apo input feature로 사용 | deployment-unavailable information leakage |
| protein-level split만 보고 pocket OOD 주장 | local site similarity leakage 가능 |
| fast throughput만 보고 ensemble method와 동일 product로 비교 | output information content가 다름 |

---

## 24. 가장 중요한 architecture lesson

이 논문을 6개월 뒤에도 기억할 이유는 BackFlip이라는 이름보다 다음 설계 원칙입니다.

### 24.1 Target symmetry를 먼저 정의한다

Output이 scalar인지 vector인지 tensor인지 먼저 정합니다.

### 24.2 Physical validity를 parameterization으로 보장한다

$$
AA^\top
$$

만으로 covariance의 SPSD constraint를 만족시킵니다.

### 24.3 Local frame에서 예측하고 known transform으로 globalize한다

복잡한 tensor network 없이도

$$
R_i\Sigma_i^{\mathrm{local}}R_i^\top
$$

으로 exact equivariance를 만들 수 있습니다.

### 24.4 Expensive process의 모든 microstate를 생성하지 않아도 된다

Downstream에 필요한 것이 descriptor라면

$$
\text{expensive trajectory}
\rightarrow
\text{summary target}
\rightarrow
\text{amortized predictor}
$$

가 훨씬 효율적일 수 있습니다.

이 마지막 원칙은 MD뿐 아니라 docking ensembles, conformer ensembles, perturbation response에서도 반복해서 쓸 수 있습니다.

---

## 25. Final verdict

**Verdict: Must Read for protein representation / geometric learning; promising but unproven as an SBDD dynamics feature.**

BackFlip-2의 가장 좋은 점은 `MD를 AI로 대체했다`는 과장된 framing이 아니라, **어떤 dynamics information을 cheap surrogate로 만들 것인지 output space를 명확하게 선택했다**는 것입니다.

RMSF 하나가 아니라 covariance tensor와 DCCM을 예측함으로써

$$
\text{how much}
+
\text{which direction}
+
\text{with whom}
$$

을 분리합니다.

그리고 covariance를 residue-local frame에서 SPSD로 만든 뒤 global frame으로 transport하는 head는 다른 geometric prediction task에도 재사용하기 좋은 pattern입니다.

SBDD 관점에서는 당장 main docking architecture에 넣기보다 **cached protein-side dynamics annotation**으로 실험하는 것이 가장 합리적입니다. 먼저 scalar RMSF → anisotropy invariants → full covariance → DCCM 순서의 matched ablation을 하고, MD-derived descriptor upper bound와 apo/holo pocket-direction test를 같이 두는 것이 좋습니다.

---

## 6개월 뒤 기억해야 할 세 가지

1. **RMSF는 covariance의 trace일 뿐이다.** BackFlip-2의 핵심은 amplitude뿐 아니라 anisotropic direction과 residue-pair coupling을 static structure에서 예측하는 것입니다.
2. **Equivariance는 이 task에서 decoration이 아니다.** Tensor target이 회전에 따라 변해야 하므로 local covariance를 global frame으로 transport하는 transformation contract가 directional accuracy를 직접 좌우합니다.
3. **이 모델은 ensemble generator가 아니라 descriptor surrogate다.** Sub-microsecond/Gaussian second-order dynamics에는 매우 빠르고 유용할 수 있지만, rare transitions, multimodal states, ligand binding utility는 별도 evidence가 필요합니다.

## Related Notes

- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[papers/analysis/benchmark-card|Benchmark card]]

## Sources

- Viliuga V. et al., **Predicting directional flexibility in proteins**, arXiv:2609.08474 — https://arxiv.org/abs/2609.08474
- Official BackFlip repository — https://github.com/graeter-group/backflip
- BackFlip repository snapshot inspected for this note — https://github.com/graeter-group/backflip/tree/887c6a73b216e3fe2ff9f551ae6e2c983c21b292
- ATLAS / mdCATH dataset and checkpoint instructions — official BackFlip README
- Paper license — CC BY 4.0 as declared by arXiv
