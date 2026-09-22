---
title: LDDM — A Unified 3D Generative Model for Synthesizable Structure-Based Drug Design
aliases:
  - papers/lddm
tags:
  - papers
  - sbdd
  - structure-based-modeling
  - molecular-generation
  - docking
  - fragment-design
  - flow-matching
  - synthesizable-design
  - protein-ligand
status: full-note
source_type: Preprint
source_url: https://doi.org/10.64898/2026.09.15.751537
---

# LDDM: A Unified 3D Generative Model for Synthesizable Structure-Based Drug Design

> **한 줄 요약:** LDDM에서 가장 재사용 가치가 높은 아이디어는 docking, fragment growing/linking, partial docking, de novo design을 별도 모델로 나누지 않고 **같은 pocket-conditioned 3D generative model에서 “무엇을 고정하고 무엇을 생성할지”만 바꾸는 conditional generation problem으로 통일**한 것이다. 여기에 reaction/building-block chemical space를 generation loop 안에 넣어 synthesizability를 post-hoc filter가 아니라 search constraint로 다루고, 저자 보고 prospective experiments와 공개 X-ray structures까지 연결했다.

## 왜 이 논문을 저장하는가

Structure-based drug design pipeline은 보통 작업별로 쪼개져 있다.

```text
known ligand
   ↓
docking model
   ↓
pose refinement
   ↓
fragment growing / linker design
   ↓
de novo generator
   ↓
synthetic accessibility filter
   ↓
virtual screening / rescoring
```

각 단계는 자연스럽지만 representation과 objective가 계속 바뀐다. Docking은 molecular graph를 고정한 채 coordinates를 찾고, de novo generation은 graph와 coordinates를 함께 만들어야 하며, fragment design은 일부 substructure만 고정한다. 그 결과 같은 protein pocket을 대상으로 하면서도 서로 다른 model family와 data pipeline을 유지하는 경우가 많다.

LDDM은 이 경계를 하나의 상태공간으로 다시 쓴다.

$$
x=(X,A,B),
$$

where

- $X\in\mathbb{R}^{N\times 3}$: ligand atom coordinates,
- $A$: atom identity / categorical state,
- $B$: covalent-bond categorical state.

그리고 task를 model identity가 아니라 **known/unknown mask**로 정의한다.

$$
\text{task}
\approx
\text{which components of }(X,A,B)\text{ are conditioned vs generated}.
$$

이 관점에서는 다음이 같은 family가 된다.

- docking: graph $(A,B)$는 known, $X$를 생성
- partial docking: 일부 $X$만 생성
- fragment growing: context fragment는 known, 나머지 $(X,A,B)$를 생성
- de novo design: ligand state 대부분을 생성

이것이 첫 번째 저장 이유다.

두 번째는 synthesizability다. 많은 3D generator가 좋은-looking molecule을 만든 뒤 SA score나 retrosynthesis로 거르는 반면, LDDM의 공개 workflow는 **reaction templates와 available building blocks가 정의하는 virtual chemical space 안에서 design action을 수행**한다. 즉

$$
\text{generate} \rightarrow \text{filter synthesizability}
$$

가 아니라

$$
\text{synthesizable action space} \rightarrow \text{generate/search}
$$

로 문제를 바꾼다.

세 번째는 evaluation이다. bioRxiv abstract 기준 저자들은 다섯 therapeutic targets에서 generated/optimized ligands를 prospective하게 시험했고, selected designs에 NMR/X-ray structural characterization을 포함했다고 보고한다. 또한 이 paper와 연결된 PDB entries가 실제 공개되어 있다. 이것은 computational benchmark와 experimental evidence가 같은 것이 아니라는 점을 분리해서 읽을 수 있는 좋은 사례다.

---

## Metadata and public artifacts

| Field | Value |
| --- | --- |
| Paper | A Unified 3D Generative Model for Synthesizable Structure-Based Drug Design |
| Authors | Ilia Igashov, Arne Schneuing, Adrian W. Dobbelstein, et al. |
| Source | bioRxiv preprint |
| Posted | 2026-09-18 |
| DOI | [10.64898/2026.09.15.751537](https://doi.org/10.64898/2026.09.15.751537) |
| Model | LDDM — Large Drug Discovery Model |
| Main task surface | docking, partial docking, fragment growing/linking, de novo design, programmable design, synthesizable design |
| Official code | [LPDI-EPFL/lddm](https://github.com/LPDI-EPFL/lddm) |
| Code snapshot inspected | [`f254fb4f8525b3803e79eb95e9f1a163fe8b2459`](https://github.com/LPDI-EPFL/lddm/commit/f254fb4f8525b3803e79eb95e9f1a163fe8b2459) |
| Code license | MIT |
| Public checkpoints/data | [Zenodo 22754501](https://zenodo.org/records/22754501) |
| Paper experiment checkpoint | `CD+BB+BN`, CC BY-NC 4.0 because it uses BindingNet |
| Alternative checkpoint | `CD+BB`, MIT, trained without BindingNet |
| Public full-scale synthesis-space caveat | Enamine REAL reactions/building blocks require a separate license and are not redistributed |
| Independent structural artifact | [RCSB PDB 38HB](https://www.rcsb.org/structure/38HB), among deposited structures linked to the paper |

> **Claim boundary:** prospective hit rates, affinities, pose accuracy, and case-study outcomes are author-reported results unless an independent public artifact is explicitly identified. A released PDB structure independently establishes that an experimental structure was deposited; it does not by itself validate every generative-model or campaign-level claim in the paper.

---

## Visual guide — 두 그림과 하나의 experimental artifact로 전체 story를 잡기

직접적인 paper-figure 재배포 대신, 공개 official repository에 포함된 MIT-licensed documentation figures를 commit-pinned URL로 사용한다. 둘 다 model/project authors가 만든 설명 자료이므로 **author-produced evidence/illustration**로 읽어야 한다.

### Official project overview — one model, many SBDD modes

![LDDM project overview](https://raw.githubusercontent.com/LPDI-EPFL/lddm/f254fb4f8525b3803e79eb95e9f1a163fe8b2459/docs/lddm.png)

*Source: LPDI-EPFL/lddm, `docs/lddm.png`, commit `f254fb4f…`, MIT license. 이 그림에서 볼 핵심은 서로 다른 downstream task가 별도 checkpoint가 아니라 동일한 pocket-conditioned generative surface에서 condition/mask를 달리해 표현된다는 점이다. 그림은 architecture concept를 설명하지만 benchmark superiority를 독립적으로 증명하지 않는다.*

### Official synthesizable-design workflow — reaction space가 action space가 된다

![LDDM synthesizable design workflow](https://raw.githubusercontent.com/LPDI-EPFL/lddm/f254fb4f8525b3803e79eb95e9f1a163fe8b2459/docs/synthgen.png)

*Source: LPDI-EPFL/lddm, `docs/synthgen.png`, commit `f254fb4f…`, MIT license. Post-hoc synthetic-accessibility score만 붙이는 대신 reaction templates와 building blocks로 실제 도달 가능한 chemical-space move를 정의한다는 점을 봐야 한다. Public demo space는 full prospective Enamine REAL campaign과 동일하지 않다.*

### Experimental structure artifact — PDB 38HB

[RCSB PDB 38HB — SARS-CoV-2 NSP3 macrodomain complex](https://www.rcsb.org/structure/38HB)

RCSB는 38HB를 2026-09-09 공개된 0.97 Å X-ray structure로 기록하며, 해당 entry의 literature를 이 LDDM paper와 연결한다. Fraser Lab publication page는 이 paper와 함께 38HB, 38HC, 38HY, 9SLI를 deposited structures로 나열한다.

이 artifact가 지지하는 범위는 명확하다.

$$
\text{public experimental complex structure exists}
$$

이지,

$$
\text{all generated poses / all targets / all hit-rate claims are independently verified}
$$

가 아니다.

---

## 1. Problem: docking과 generation은 정말 다른 문제인가

Docking을 가장 단순하게 쓰면:

$$
\hat X
=
\arg\max_X p(X\mid A,B,P),
$$

where $P$ is the protein pocket.

De novo design은:

$$
(\hat X,\hat A,\hat B)
\sim
p(X,A,B\mid P).
$$

Fragment growing에서는 ligand 일부 $C$가 고정된다.

$$
(X_G,A_G,B_G)
\sim
p(X_G,A_G,B_G\mid P,C).
$$

표면적으로 output space가 다르지만, 세 문제 모두 **partial observation이 주어진 structured molecular state completion**으로 볼 수 있다.

LDDM의 핵심 abstraction은 여기다.

> Docking과 generation을 architecture-level task label로 분리하기보다, molecular state의 어떤 variable이 known이고 어떤 variable이 unknown인지로 분리한다.

이렇게 하면 constrained design도 자연스럽다. Scaffold, warhead, anchor atom, reference fragment처럼 유지해야 하는 요소는 known context로 남기고 나머지만 stochastic generation에 넣을 수 있다.

---

## 2. Representation contract: molecule은 graph + coordinates의 joint state다

3D molecular model에서 자주 생기는 오류는 geometry와 chemistry를 한쪽에 종속시키는 것이다.

- coordinates만 생성하고 molecular graph를 후처리로 추론하거나,
- 2D graph를 먼저 만든 뒤 conformer를 별도 생성하거나,
- docking에서는 graph를 fixed input으로만 취급한다.

LDDM은 공개 implementation에서 continuous state와 discrete state를 함께 다루는 components를 갖는다.

### Continuous state

Coordinates는 flow-matching formulation으로 이동한다. 공개 code의 `RiemannianICFM` interface는 noisy/intermediate state $z_t$를 만들고, network prediction에서 vector field를 복원해 ODE-style update를 수행한다.

Linear schedule intuition에서는:

$$
z_t=(1-t)z_0+t z_1,
$$

이고 target velocity는 개념적으로

$$
v^*(z_t,t)
\propto
\frac{z_1-z_t}{1-t}.
$$

Model은 이 vector field 또는 final-state-related target을 학습하고 inference에서 적분한다.

### Discrete state

Atom type과 bond category는 continuous coordinate처럼 단순 보간할 수 없다. 공개 code는 Markov-bridge transition을 사용한다.

Uniform-prior bridge에서 transition은:

$$
Q_t
=
\beta_t I
+
(1-\beta_t)\mathbf 1 z_1^\top.
$$

Network가 final categorical state의 distribution을 예측하면 현재 discrete state에서 다음 state를 sample한다.

따라서 LDDM의 generative object는 단순 point cloud가 아니다.

$$
\boxed{
\text{molecular state}
=
\text{continuous geometry}
+
\text{discrete chemistry}
}
$$

이 joint-state view가 docking과 de novo design을 같은 model에 넣을 수 있게 한다.

---

## 3. Symmetry: protein pocket의 frame이 바뀌면 output도 같이 움직여야 한다

Pocket-conditioned coordinate generation은 rigid-frame choice에 민감하면 안 된다.

Protein/ligand coordinates에 rigid transform

$$
x' = Rx+t,
\qquad R\in SO(3)
$$

를 적용했을 때 generated coordinate update도 같은 방식으로 변환되어야 한다.

$$
v(RX+t,RP+t,t)
=
R\,v(X,P,t).
$$

반면 atom identity나 bond probability는 rotation에 대해 invariant해야 한다.

공개 source tree에 GVP와 heterogeneous geometric GNN components가 포함되어 있고 model은 coordinate flow를 graph-conditioned dynamics로 구현한다. 실무적으로 중요한 것은 architecture 이름보다 **coordinate target은 equivariant, chemistry logits는 invariant**라는 output contract다.

이 contract가 깨지면 arbitrary alignment나 reference-ligand frame을 통해 성능이 부풀려질 수 있다. 따라서 LDDM류 모델을 다른 benchmark로 옮길 때는 pocket extraction과 centering/alignment가 deployment에서 사용 가능한 정보만 쓰는지도 같이 감사해야 한다.

---

## 4. Unified task masking: 같은 checkpoint가 mode를 바꾸는 원리

Official README에서 같은 checkpoint를 `design`과 `dock` mode에서 사용한다. Partial docking은 `--atoms_to_dock`으로 생성할 atom subset을 지정한다.

이를 mask notation으로 쓰면:

$$
m_X,m_A,m_B\in\{0,1\}
$$

이고, $m=1$을 generate, $m=0$을 condition/fix라고 두자.

### Docking

$$
m_A=0,\qquad m_B=0,\qquad m_X=1.
$$

Chemical identity는 고정하고 pose만 생성한다.

### Partial docking

$$
m_{X,i}=1 \quad \text{only for selected atoms}.
$$

나머지 atoms는 anchor/context로 남는다. Covalent or anchored-ligand problem에 특히 자연스럽다.

### Fragment growing / linking

Known fragment에 속한 variables는 conditioning context가 되고 나머지 fragment variables를 생성한다.

### De novo design

$$
m_X\approx m_A\approx m_B\approx 1.
$$

Pocket을 제외한 ligand state를 크게 열어 둔다.

이 unified masking 관점은 단순 software convenience 이상이다. 학습 representation이

$$
\text{pose completion}
\leftrightarrow
\text{chemical completion}
$$

사이에서 공유되기 때문에, docking에서 배운 protein–ligand geometry와 design에서 배운 chemical completion이 같은 latent dynamics 안에서 만날 수 있다.

다만 이것이 실제 positive transfer를 자동으로 보장하지는 않는다. Multi-task sharing이 interference를 만들 수도 있기 때문에 matched single-task baselines가 필요하다.

---

## 5. Sampling contract: quality는 model뿐 아니라 integration budget의 함수다

Official repository의 기본 sampling 설정은:

- `n_steps = 100`
- ODE sampler: `ForwardEuler` or `HeunSampler`
- coordinate sampling noise configurable
- `n_samples` configurable

이다.

따라서 pose/design 결과를 비교할 때 반드시 다음을 같이 기록해야 한다.

$$
\text{checkpoint}
+
\text{sampler}
+
\text{integration steps}
+
\text{noise}
+
\text{number of samples}
+
\text{selection rule}.
$$

Top-1 pose와 best-of-$K$ pose는 다른 claim이다. Generator끼리 비교하면서 한쪽은 10 samples, 다른 쪽은 100 samples를 만들고 oracle-like selection을 사용하면 architecture 비교가 아니다.

특히 prospective design에서는 generated pool → programmable filtering/search → synthesis selection이 이어지므로 최종 wet-lab hit rate는 base model뿐 아니라 **sampling and selection policy 전체**의 결과다.

---

## 6. Programmable design: generator를 optimizer의 proposal distribution으로 쓴다

LDDM의 중요한 확장은 one-shot conditional generation에서 끝나지 않는다는 점이다.

Programmable design은 generated candidates를 평가하고 다시 design action을 취하는 iterative loop로 볼 수 있다.

$$
x_{k+1}
\sim
q_\theta(\cdot\mid P,x_k,c_k),
$$

where $c_k$ encodes fixed context or local design constraints.

그리고 evaluator가 candidate utility를 계산한다.

$$
u_k
=
U(x_k,P).
$$

이때 model은 최종 optimizer 자체라기보다 **chemistry-aware proposal operator**가 된다.

이 구분은 중요하다.

- generative prior가 좋은가?
- evaluator가 좋은 candidate를 잘 구분하는가?
- search policy가 compute budget을 잘 배분하는가?

를 따로 측정할 수 있기 때문이다.

[[papers/sbdd/adaptiveflow|AdaptiveFlow]]가 ultra-large library에서 `어떤 molecule을 oracle에 보낼 것인가`를 최적화한다면, LDDM은 local/generated chemical state에서 `어떤 chemically meaningful modification을 제안할 것인가`를 learned generator로 수행한다고 볼 수 있다.

---

## 7. Synthesizable design: SA score가 아니라 reaction graph 위의 generation

가장 실용적으로 흥미로운 부분이다.

많은 generative pipeline은:

$$
x\sim p_\theta(x\mid P)
\quad\rightarrow\quad
\operatorname{SA}(x)\text{ filter}
$$

를 사용한다.

하지만 낮은 SA score가 실제 available building blocks와 reaction route를 보장하지는 않는다.

LDDM public workflow는 다음을 입력으로 받는다.

- building-block table
- reaction SMARTS/templates
- reaction-to-building-block role mapping

따라서 design move 자체가 reaction-valid chemical space 안에서 정의될 수 있다.

$$
\mathcal X_{synth}
=
\{R(b_i,b_j): R\in\mathcal R,\ b_i,b_j\in\mathcal B\}.
$$

여기서 $\mathcal R$은 reaction set, $\mathcal B$는 available building blocks다.

핵심은:

$$
\text{synthetic feasibility}
\approx
\text{generation-domain constraint}
$$

로 들어간다는 것이다.

### 재현성 경계

Official repo는 full prospective Enamine REAL reactions/building blocks를 license 때문에 배포하지 않는다. 대신 public demo로 SynSpace-derived:

- 44,944 building blocks
- 3 reactions

을 제공한다.

따라서 code를 실행해 synthesizable workflow를 확인하는 것과 **paper의 full prospective search space를 그대로 재현하는 것**은 다르다.

이 차이는 reproducibility note에서 반드시 남겨야 한다.

---

## 8. Evaluation을 네 층으로 분리해야 한다

LDDM paper의 headline은 강하지만 evidence를 한 숫자로 합치면 안 된다.

### Layer A — geometric / docking evidence

질문:

> 주어진 ligand graph에 대해 native-like pose를 생성할 수 있는가?

적절한 metric은 symmetry-aware pose RMSD, validity, clash/strain 등이다.

이 evidence는 affinity를 직접 증명하지 않는다.

### Layer B — generative chemistry evidence

질문:

> pocket-conditioned generation이 chemically valid하고 diverse하며 target-relevant한 candidates를 만드는가?

Validity/diversity/novelty와 pocket geometry가 필요하다. Synthetic accessibility는 또 별도다.

### Layer C — prospective binding evidence

질문:

> 실제 합성해서 측정했을 때 binding/activity가 확인되는가?

bioRxiv abstract는 다섯 therapeutic targets에서 prospective validation을 했고 모든 case에서 소수 합성으로 confirmed binders를 확보했다고 저자들이 보고한다고 명시한다.

이것은 retrospective docking benchmark보다 강한 deployment-facing evidence지만 campaign별 synthesis budget, selection policy, assay, threshold가 결과 해석에 포함되어야 한다.

### Layer D — structural validation

질문:

> selected ligand의 experimentally observed binding geometry가 proposed pose와 일치하는가?

Paper는 best designs 일부를 NMR spectroscopy와 X-ray crystallography로 characterize했다고 보고한다. 공개 artifact로는 Fraser Lab page가 38HB, 38HC, 38HY, 9SLI를 이 paper와 연결하고, RCSB의 38HB는 0.97 Å X-ray complex로 공개되어 있다.

이 layer는 pose claim에 매우 유용하지만 selected-success cases의 structure가 전체 generated distribution의 unbiased estimate는 아니다.

---

## 9. 무엇이 실제 novelty인가

LDDM의 contribution을 세 층으로 분리하면 명확하다.

### A. Unified conditional state completion

Docking, partial docking, fragment editing, de novo design을 같은 $(X,A,B)$ state와 mask로 표현한다.

### B. Mixed continuous–discrete generation

Coordinates는 flow-matching dynamics로, atom/bond chemistry는 discrete Markov bridge로 함께 생성한다.

### C. Synthesizable programmable search

Generator를 one-shot outputter가 아니라 constrained proposal operator로 사용하고, reaction/building-block space를 search domain으로 넣는다.

Prospective experiments는 이 세 layer가 실제 discovery workflow로 이어질 수 있다는 system-level evidence를 제공하지만, 각 component의 causal contribution을 따로 증명하는 것은 아니다.

---

## 10. Baseline fairness에서 가장 먼저 볼 것

Unified model은 task count가 많기 때문에 baseline 비교가 특히 어렵다.

### Docking comparison

맞춰야 하는 것:

- receptor state
- pocket definition
- protonation/tautomer state
- reference-ligand information
- number of generated poses
- sampling steps
- ranking/selection method
- symmetry treatment

### Generation comparison

맞춰야 하는 것:

- allowed molecule size distribution
- scaffold/fragment constraints
- sample count
- property filters
- validity criteria
- synthetic-space constraints
- compute/oracle calls

### Prospective comparison

가장 중요하지만 가장 어렵다.

$$
\text{hit rate}
=
f(\text{generator},\text{search},\text{filter},\text{chemist selection},\text{assay},\text{synthesis budget}).
$$

따라서 prospective hit rate를 raw model score로 읽으면 안 된다.

---

## 11. Split와 OOD boundary

Protein–ligand generative model의 generalization claim은 molecule split 하나로 끝나지 않는다.

적어도 다음 축이 있다.

| Axis | Possible leakage / interpolation |
| --- | --- |
| Ligand | close scaffold, analog series, stereoisomer |
| Protein | close sequence/structure family |
| Complex | near-identical pocket–ligand interaction pattern |
| Time | training cutoff 이후의 new structures/assays인가? |
| Pocket state | holo/apo/conformational-state shift |
| Chemistry | building blocks/reaction templates seen during training/search |
| Selection | test-target-specific tuning or evaluator adaptation |

특히 docking과 design을 한 checkpoint에서 학습하면 같은 complex가 서로 다른 masking task로 노출될 수 있다. Example-level split가 아니라 **underlying complex/family/scaffold identity 단위**로 분리됐는지를 확인해야 한다.

Prospective wet-lab validation은 retrospective leakage 우려를 크게 줄여 주지만, 그것만으로 broad OOD generalization을 증명하지는 않는다. 다섯 selected targets에서의 success는 그 campaign scope의 evidence다.

---

## 12. Reference-ligand and pocket-definition leakage

Official examples는 protein과 함께 `--ref_ligand`를 전달한다. 이것은 pocket definition이나 coordinate reference를 제공하는 practical interface일 수 있다.

하지만 benchmark claim에서는 다음을 분리해야 한다.

1. reference ligand가 **pocket localization만** 제공하는가?
2. reference ligand의 exact pose/shape가 generative input에 들어가는가?
3. evaluated ligand 자체 또는 close analog information이 들어가는가?
4. deployment에서 같은 information이 실제로 available한가?

Known holo ligand를 이용한 site-defined generation과 apo/unliganded target discovery는 다른 task다.

따라서 reference-ligand availability를 숨긴 채 “protein-only de novo design”으로 일반화하면 안 된다.

---

## 13. Reproducibility: public surface는 강하지만 완전하지 않다

Public artifact 측면에서 이 paper는 상당히 좋은 편이다.

확인 가능한 것은:

- MIT-licensed source code
- executable docking/design examples
- Docker environment
- main paper checkpoint
- non-BindingNet checkpoint
- geometry reference data
- public SynSpace-derived synthesizable-demo space
- configurable ODE sampling
- reaction-space preparation scripts

이다.

하지만 두 가지 licensing boundary가 있다.

### Main checkpoint

`CD+BB+BN`은 paper experiments에 사용된 checkpoint이며 BindingNet 때문에 CC BY-NC 4.0이다.

### Full synthesis space

Enamine REAL reactions/building blocks는 별도 license가 필요하고 repo에서 재배포되지 않는다.

따라서 공개 artifact가 충분하다는 것과 **paper의 모든 prospective experiment를 byte-for-byte 재현할 수 있다**는 것은 다르다.

재현 시 최소 pinning contract는:

```text
paper version
+ repository commit
+ checkpoint identity/license
+ protein/pocket preparation
+ reference ligand role
+ sampler + n_steps + noise
+ n_samples
+ candidate selection rule
+ chemical-space release
```

여야 한다.

---

## 14. 가장 중요한 confounders

### 14.1 Unified-model gain vs data gain

여러 task를 한 model에서 학습하면 더 많은 heterogeneous supervision을 사용하게 된다. 성능 gain이 **masking/unification 자체** 때문인지 단순 data volume/coverage 때문인지 matched single-task control이 필요하다.

### 14.2 Generator vs search/evaluator

Prospective candidate는 base generator output을 그대로 무작위 합성한 것이 아니다. Programmable search와 evaluator/filters가 개입한다. 따라서 wet-lab result를 base model likelihood의 직접 측정값으로 읽으면 안 된다.

### 14.3 Synthesizability vs purchasability

Reaction-template reachable은 practical chemistry에 가까운 constraint지만, yield, conditions, protecting groups, vendor availability, cost, purification까지 보장하지는 않는다.

### 14.4 Structural confirmation selection bias

NMR/X-ray가 있는 성공 사례는 매우 가치 있지만 보통 selected hits에 집중된다. 이는 pose mechanism evidence이지 전체 generated set의 calibration curve가 아니다.

### 14.5 Preprint status

2026-09-18 공개된 bioRxiv preprint이므로 peer-review 과정의 수정 가능성이 있다. Version/date를 기록해야 한다.

---

## 15. Decision-useful ablations

이 model을 제대로 이해하려면 headline benchmark보다 다음 controls가 더 중요하다.

### Experiment A — unified vs task-specific model

동일 training examples와 total compute에서:

```text
A0: docking-only model
A1: design-only model
A2: unified masked model
```

을 비교한다.

질문은 단순 평균 성능이 아니다.

- docking supervision이 de novo design에 positive transfer를 주는가?
- design supervision이 pose quality를 개선하는가?
- task interference는 어디에서 생기는가?

### Experiment B — partial-state curriculum

Mask distribution을 바꾼다.

```text
B0: full docking + full design only
B1: + fragment growing/linking masks
B2: + random/local partial-coordinate masks
```

이렇게 하면 partial completion이 genuinely reusable representation을 만드는지 볼 수 있다.

### Experiment C — synthesis-aware search vs generate-then-filter

동일한 total generator calls / scoring calls / wall-clock budget에서:

```text
C0: unconstrained generation → SA/retro filter
C1: reaction-constrained generation/search
```

를 비교한다.

평가는:

- final valid candidates
- chemical diversity
- target utility
- reaction-route validity
- available-building-block coverage
- oracle calls per accepted candidate

를 함께 봐야 한다.

### Experiment D — reference-ligand dependence

```text
D0: known holo reference ligand
D1: pocket center only
D2: apo structure / predicted pocket
```

를 비교하면 deployment boundary가 드러난다.

### Experiment E — sampling budget curve

$K\in\{1,5,10,50,100\}$ samples에서:

- top-ranked pose quality
- oracle pose quality
- diversity
- validity
- compute

를 같이 기록한다. Generator capacity와 selector quality를 분리할 수 있다.

---

## 16. Falsification: 어떤 결과가 나오면 unified story를 약하게 봐야 하나

강한 architecture claim은 반증 조건이 있어야 한다.

다음 결과가 나오면 “one model for all SBDD tasks”의 mechanistic advantage는 약해진다.

1. 같은 data/compute에서 task-specific models가 모든 주요 task에서 일관되게 우세하다.
2. Unified model gain이 단순 training-set enlargement control에서 사라진다.
3. Partial masks를 제거해도 fragment/docking transfer 성능이 변하지 않는다.
4. Reaction-constrained search가 matched compute에서 generate-then-filter보다 quality–diversity–synthesizability Pareto를 개선하지 못한다.
5. Reference ligand를 제거하면 성능이 급락하고 pocket-only setup에서 회복되지 않는다.
6. Prospective success가 다른 protein-family / assay context에서 재현되지 않는다.

반대로 이런 controls를 통과한다면 LDDM의 진짜 contribution은 특정 leaderboard score보다 **partial-state molecular generation이라는 reusable SBDD interface**라고 볼 근거가 강해진다.

---

## 17. 실무적으로 가장 흥미로운 연구 방향

LDDM이 제시하는 가장 유용한 질문은 “하나의 거대 generator가 모든 것을 해야 하는가?”가 아니다.

더 좋은 질문은:

> **Dock → refine → grow → redesign을 서로 다른 representation으로 넘기지 않고, 하나의 structured molecular state에서 uncertainty와 constraints를 유지한 채 연속적으로 수행할 수 있는가?**

이를 generic SBDD pipeline으로 쓰면:

```text
protein pocket + partial ligand state
             ↓
    joint 3D/chemical generator
             ↓
 pose / partial edit / new chemistry
             ↓
 independent validity + scoring
             ↓
 reaction-aware constrained search
             ↓
 final candidates
```

가 된다.

여기서 independent scoring을 남겨 두는 것이 중요하다. Generator 자체의 confidence와 같은 model family의 evaluator만 사용하면 correlated error가 selection loop에서 증폭될 수 있다.

따라서 실제 연구에서는 **unified proposal model + independent verifier** 조합이 더 안전한 baseline이다.

---

## 18. Pose generation과 affinity prediction을 섞지 말 것

LDDM의 주요 contribution은 generative structure/design 쪽이다.

Native-like pose를 잘 만든다는 것은:

$$
p(X\mid A,B,P)
$$

를 잘 모델링한다는 evidence다.

Binding affinity는:

$$
\Delta G_{bind}
$$

혹은 assay-specific potency를 예측/최적화하는 다른 target이다.

Pose quality가 좋아도:

- desolvation
- entropy
- protonation changes
- water networks
- receptor reorganization
- assay context

를 충분히 모델링하지 않으면 affinity ranking은 틀릴 수 있다.

따라서 prospective binder discovery가 성공했다고 해서 generator가 calibrated affinity predictor라는 결론은 나오지 않는다.

이 구분은 [[concepts/sbdd/pose-quality|Pose quality]]와 [[concepts/sbdd/binding-affinity|Binding affinity]]를 같이 읽어야 하는 이유다.

---

## 19. Computational benchmark보다 prospective evidence를 어떻게 읽을까

Prospective experiment는 retrospective split보다 leakage-resistant하다는 큰 장점이 있다. 하지만 prospective라는 단어만으로 공정성이 자동 보장되지는 않는다.

Decision-useful campaign card에는 적어도:

- target identity / novelty relative to training
- starting ligand or fragment availability
- generated count
- filtered/scored count
- human selection 여부
- synthesized count
- assay endpoint
- hit threshold
- confirmed hit count
- structural follow-up selection rule

가 필요하다.

BioRxiv abstract는 `five targets`, `small number synthesized`, `confirmed binding`, `NMR/X-ray on best designs`라는 강한 high-level evidence를 제공한다. 하지만 target별 exact selection budget과 affinity distribution을 모두 abstract에서 읽을 수는 없다.

따라서 이 note의 verdict는 **prospective evidence가 존재한다**는 것과 **모든 campaign efficiency 수치가 독립 재현되었다**는 것을 의도적으로 분리한다.

---

## 20. Related notes

- [[papers/sbdd/posebusters|PoseBusters]] — generated/docked pose가 chemically and geometrically plausible한지 보는 validity boundary
- [[papers/sbdd/adaptiveflow|AdaptiveFlow]] — expensive oracle 아래 compute budget과 candidate-selection policy를 최적화하는 complementary view
- [[papers/generative-models/ensemble-conditioned-molecular-design|Ensemble-Conditioned Molecular Design]] — inference-time condition composition으로 multi-state target/avoid objective를 만드는 다른 conditional-generation route
- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]

특히 LDDM과 AdaptiveFlow를 같이 보면 두 종류의 search problem이 분리된다.

$$
\text{AdaptiveFlow}: \text{which existing molecules should be evaluated?}
$$

$$
\text{LDDM}: \text{which new molecular state/change should be proposed?}
$$

둘 다 model architecture만큼 **allocation/search policy**가 중요하다.

---

## Final verdict

LDDM의 headline은 “3D molecule generator가 prospective hits를 냈다”이지만, 오래 남는 contribution은 더 구조적이다.

첫째, docking과 de novo design 사이의 경계를 없애고 **partial molecular state completion**이라는 하나의 interface로 다시 썼다. Molecular graph와 coordinates를 joint state로 두면 pose-only, partial-coordinate, fragment, full-generation task를 mask만 바꿔 표현할 수 있다.

둘째, synthesizability를 scalar filter로만 보지 않고 **reaction/building-block-defined action space**로 바꿨다. 이것은 생성 quality뿐 아니라 실제 search efficiency와 실험 가능성을 architecture 주변의 first-class objective로 만든다.

셋째, prospective binding과 public structural artifacts가 있어 benchmark-only generative paper보다 evidence가 강하다. 동시에 full Enamine REAL space의 licensing, main checkpoint의 non-commercial license, selection/search confound, target별 experimental-budget detail 같은 reproducibility boundary가 분명히 남아 있다.

따라서 이 paper를 인용할 때 가장 안전한 문장은 다음과 같다.

> **LDDM은 하나의 pocket-conditioned mixed continuous–discrete generator를 masking/conditioning으로 재사용해 docking부터 molecular design까지 여러 SBDD task를 통합하고, reaction-space-constrained search와 prospective experiments를 통해 이 unified formulation의 practical potential을 보여준다.**

`모든 SBDD task에서 최적이다`, `affinity를 정확히 예측한다`, `full prospective campaign이 완전히 공개 재현 가능하다`는 더 강한 주장은 현재 evidence와 분리해야 한다.

---

## Three durable takeaways

1. **Docking과 molecular generation은 partial-state completion의 서로 다른 mask로 통일할 수 있다.**  
   Graph, bond, coordinate를 joint state로 두면 dock/grow/link/de novo를 같은 model surface에서 표현할 수 있고, 실제 연구 질문은 task sharing이 positive transfer를 만드는지 matched ablation으로 확인하는 것이다.

2. **Synthesizability는 post-hoc score보다 action-space constraint로 넣을 때 더 강한 의미를 갖는다.**  
   Reaction templates와 available building blocks가 generation/search domain을 정의하면 “만들 수 있을 법한 molecule”이 아니라 “주어진 chemical space에서 구성 가능한 molecule” 쪽으로 objective가 이동한다.

3. **Prospective evidence도 generator, selector, chemistry space, assay를 분리해서 읽어야 한다.**  
   Confirmed binders와 NMR/X-ray structures는 중요한 evidence지만, end-to-end campaign success는 base model 하나가 아니라 sampling, programmable search, filtering, synthesis selection, assay의 합성 결과다.

---

## Sources

- Igashov I, Schneuing A, Dobbelstein AW, et al. [A Unified 3D Generative Model for Synthesizable Structure-Based Drug Design](https://doi.org/10.64898/2026.09.15.751537). bioRxiv, posted 2026-09-18.
- Official implementation: [LPDI-EPFL/lddm](https://github.com/LPDI-EPFL/lddm), inspected at commit [`f254fb4f8525b3803e79eb95e9f1a163fe8b2459`](https://github.com/LPDI-EPFL/lddm/commit/f254fb4f8525b3803e79eb95e9f1a163fe8b2459).
- Public checkpoints and reference data: [Zenodo record 22754501](https://zenodo.org/records/22754501).
- Independent experimental structure record: [RCSB PDB 38HB](https://www.rcsb.org/structure/38HB).
- Paper-linked structure list: [Fraser Lab publications](https://fraserlab.com/publications/).
