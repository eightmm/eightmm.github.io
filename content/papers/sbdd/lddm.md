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

> **한 줄 요약:** LDDM에서 가장 재사용 가치가 높은 아이디어는 docking, fragment growing/linking, partial docking, de novo design을 별도 모델로 나누지 않고 **같은 pocket-conditioned 3D generative model에서 무엇을 고정하고 무엇을 생성할지만 바꾸는 conditional generation problem으로 통일**한 것이다. 여기에 reaction/building-block chemical space를 generation loop 안에 넣어 synthesizability를 post-hoc filter가 아니라 search constraint로 다루고, 저자 보고 prospective experiments와 공개 X-ray structures까지 연결했다.

## 왜 이 논문을 저장하는가

Structure-based drug design pipeline은 보통 작업별로 쪼개진다.

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

각 단계는 자연스럽지만 representation과 objective가 계속 바뀐다. Docking은 molecular graph를 고정한 채 coordinates를 찾고, de novo generation은 graph와 coordinates를 함께 만들어야 하며, fragment design은 일부 substructure만 고정한다. 같은 protein pocket을 대상으로 하면서도 서로 다른 model family와 data pipeline을 유지하는 경우가 많다.

LDDM은 이 경계를 하나의 상태공간으로 다시 쓴다.

$$
x=(X,A,B),
$$

where $X\in\mathbb{R}^{N\times3}$ is ligand coordinates, $A$ is atom identity, and $B$ is covalent-bond state.

Task는 model identity가 아니라 **known/unknown mask**로 정의할 수 있다.

$$
\text{task}
\approx
\text{which components of }(X,A,B)\text{ are conditioned vs generated}.
$$

이 관점에서는 docking, partial docking, fragment growing/linking, de novo design이 같은 structured-state completion family가 된다. 이것이 첫 번째 저장 이유다.

두 번째는 synthesizability다. 많은 3D generator가 molecule을 만든 뒤 SA score나 retrosynthesis로 거르는 반면, LDDM의 공개 workflow는 **reaction templates와 available building blocks가 정의하는 virtual chemical space 안에서 design action을 수행**한다.

$$
\text{generate}\rightarrow\text{filter synthesizability}
$$

보다

$$
\text{synthesizable action space}\rightarrow\text{generate/search}
$$

에 가깝다.

세 번째는 evaluation이다. bioRxiv abstract 기준 저자들은 다섯 therapeutic targets에서 generated/optimized ligands를 prospective하게 시험했고, selected designs에 NMR/X-ray characterization을 포함했다고 보고한다. 이 paper와 연결된 PDB entries도 공개되어 있다. Computational benchmark와 experimental evidence를 같은 것으로 취급하지 않으면서 end-to-end design story를 볼 수 있는 좋은 사례다.

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
| Task surface | docking, partial docking, fragment growing/linking, de novo design, programmable design, synthesizable design |
| Official code | [LPDI-EPFL/lddm](https://github.com/LPDI-EPFL/lddm) |
| Code snapshot inspected | [`f254fb4f8525b3803e79eb95e9f1a163fe8b2459`](https://github.com/LPDI-EPFL/lddm/commit/f254fb4f8525b3803e79eb95e9f1a163fe8b2459) |
| Code license | MIT |
| Public checkpoints/data | [Zenodo 22754501](https://zenodo.org/records/22754501) |
| Paper checkpoint | `CD+BB+BN`, CC BY-NC 4.0 because BindingNet is included |
| Alternative checkpoint | `CD+BB`, MIT, without BindingNet |
| Synthesis-space caveat | Enamine REAL reactions/building blocks require a separate license and are not redistributed |
| Structural artifact | [RCSB PDB 38HB](https://www.rcsb.org/structure/38HB), among structures linked to the paper |

> **Claim boundary:** prospective hit rates, affinities, pose accuracy, and campaign outcomes are author-reported unless an independent artifact is explicitly identified. A public PDB entry establishes that an experimental complex structure exists; it does not independently validate every generative-model or campaign-level claim.

---

## Visual guide

직접적인 paper-figure 재배포 대신 공개 official repository의 MIT-licensed documentation figures를 commit-pinned URL로 사용한다. 두 그림 모두 author-produced illustration이므로 benchmark evidence와 구분해야 한다.

### One model, many SBDD modes

![LDDM project overview](https://raw.githubusercontent.com/LPDI-EPFL/lddm/f254fb4f8525b3803e79eb95e9f1a163fe8b2459/docs/lddm.png)

*Source: LPDI-EPFL/lddm, `docs/lddm.png`, commit `f254fb4f…`, MIT license. 핵심은 서로 다른 downstream task가 별도 checkpoint가 아니라 동일한 pocket-conditioned generative surface에서 condition/mask를 달리해 표현된다는 점이다. 이 그림은 architecture concept를 설명하며 benchmark superiority를 독립적으로 증명하지 않는다.*

### Reaction-space-constrained design

![LDDM synthesizable design workflow](https://raw.githubusercontent.com/LPDI-EPFL/lddm/f254fb4f8525b3803e79eb95e9f1a163fe8b2459/docs/synthgen.png)

*Source: LPDI-EPFL/lddm, `docs/synthgen.png`, commit `f254fb4f…`, MIT license. Post-hoc synthetic-accessibility score만 붙이는 대신 reaction templates와 building blocks로 실제 도달 가능한 chemical-space move를 정의한다는 점을 봐야 한다. Public demo space는 full prospective Enamine REAL campaign과 동일하지 않다.*

### Independent experimental artifact — PDB 38HB

[RCSB PDB 38HB — SARS-CoV-2 NSP3 macrodomain complex](https://www.rcsb.org/structure/38HB)

RCSB는 38HB를 2026-09-09 공개된 0.97 Å X-ray structure로 기록하며 해당 entry의 literature를 이 LDDM paper와 연결한다. Fraser Lab publication page는 38HB, 38HC, 38HY, 9SLI를 이 paper와 연결된 deposited structures로 나열한다.

이 artifact가 직접 지지하는 범위는

$$
\text{public experimental complex structure exists}
$$

이지

$$
\text{all generated poses and campaign claims are independently verified}
$$

가 아니다.

---

## 1. Docking과 generation은 partial-state completion의 두 모드다

Docking을 단순화하면

$$
\hat X\sim p(X\mid A,B,P),
$$

where $P$ is the protein pocket.

De novo design은

$$
(\hat X,\hat A,\hat B)\sim p(X,A,B\mid P).
$$

Fragment growing에서는 known context $C$가 추가된다.

$$
(X_G,A_G,B_G)\sim p(X_G,A_G,B_G\mid P,C).
$$

Output space가 달라 보이지만 모두 **partial observation이 주어진 molecular state completion**으로 볼 수 있다.

LDDM의 유용한 abstraction은 docking과 generation을 architecture-level task label로 분리하기보다 molecular state의 어떤 variable이 known이고 어떤 variable이 unknown인지로 분리하는 것이다. Scaffold, anchor, warhead, known fragment는 conditioning context로 남기고 나머지만 stochastic generation에 넣을 수 있다.

---

## 2. Joint representation: geometry와 chemistry를 같이 생성한다

3D molecular generation은 geometry와 chemistry 중 하나를 후처리로 미루기 쉽다. LDDM의 공개 source tree는 continuous state와 discrete state를 함께 다루는 components를 제공한다.

### Continuous coordinates

공개 `flows.py`의 `RiemannianICFM` interface는 noisy/intermediate state $z_t$를 만들고 network prediction에서 vector field를 복원해 ODE-style update를 수행한다.

Linear path intuition에서는

$$
z_t=(1-t)z_0+t z_1,
$$

이고 velocity target은 개념적으로

$$
v^*(z_t,t)\propto\frac{z_1-z_t}{1-t}.
$$

Inference에서는 learned field를 적분해 coordinate state를 이동시킨다.

### Discrete atom and bond states

Atom type과 bond category는 continuous coordinate처럼 단순 보간할 수 없다. 공개 `markov_bridge.py`는 Markov bridge transition을 구현한다.

Uniform-prior bridge에서 transition은

$$
Q_t=\beta_t I+(1-\beta_t)\mathbf 1 z_1^\top.
$$

Network가 final categorical state distribution을 예측하면 현재 discrete state에서 다음 state를 sample한다.

따라서 generative object는 단순 point cloud가 아니다.

$$
\boxed{\text{molecular state}=\text{continuous geometry}+\text{discrete chemistry}}
$$

이 joint-state view가 pose-only와 chemistry-generating tasks를 같은 model surface에 올릴 수 있게 한다.

---

## 3. Symmetry contract

Pocket-conditioned coordinate generation은 arbitrary rigid frame에 의존하면 안 된다.

$$
x'=Rx+t,\qquad R\in SO(3)
$$

로 protein/ligand coordinates를 변환했을 때 coordinate update도

$$
v(RX+t,RP+t,t)=Rv(X,P,t)
$$

처럼 변환되어야 한다. 반면 atom identity와 bond probability는 rotation에 invariant해야 한다.

Public source tree에는 GVP와 heterogeneous geometric GNN components가 포함되어 있다. Architecture 이름보다 중요한 contract는 **coordinate output은 equivariant, chemistry logits는 invariant**라는 점이다.

이 contract가 깨지면 arbitrary alignment나 reference-ligand frame을 통해 성능이 부풀려질 수 있다. 다른 benchmark로 옮길 때 pocket extraction과 centering/alignment가 deployment에서 사용 가능한 정보만 쓰는지도 같이 감사해야 한다.

---

## 4. Unified task masking

Official README에서는 같은 checkpoint를 `design`과 `dock` mode에서 사용하고, partial docking은 `--atoms_to_dock`으로 생성할 atom subset을 지정한다.

Mask를

$$
m_X,m_A,m_B\in\{0,1\}
$$

로 두고 $1$을 generate, $0$을 condition/fix라고 하자.

### Docking

$$
m_A=0,\quad m_B=0,\quad m_X=1.
$$

Chemical identity는 고정하고 pose만 생성한다.

### Partial docking

$$
m_{X,i}=1
$$

을 selected atoms에만 적용한다. 나머지는 anchor/context로 남는다.

### Fragment growing and linking

Known fragment variables는 condition이고 새 fragment의 coordinates/chemistry가 generation target이 된다.

### De novo design

$$
m_X\approx m_A\approx m_B\approx1.
$$

Pocket 이외의 ligand state를 크게 연다.

이 unified masking은 software convenience 이상이다. Pose completion과 chemical completion의 relational geometry가 같은 latent dynamics를 공유할 수 있기 때문이다. 다만 positive transfer가 자동으로 보장되는 것은 아니다. Multi-task interference가 있을 수 있으므로 matched single-task baseline이 필요하다.

---

## 5. Sampling contract

Official repository의 기본 sampling interface는 `n_steps=100`이며 Forward Euler 또는 Heun sampler를 지원한다. Coordinate sampling noise와 number of samples도 설정할 수 있다.

따라서 결과 비교에는 최소한

$$
\text{checkpoint}+
\text{sampler}+
\text{steps}+
\text{noise}+
K+\text{selection rule}
$$

을 같이 기록해야 한다.

Top-1과 best-of-$K$는 다른 claim이다. 한 method가 10 poses를 만들고 다른 method가 100 poses를 만든 뒤 near-oracle selection을 사용한다면 architecture comparison이 아니다.

Prospective design에서는 generated pool → programmable search/filter → synthesis selection이 이어지므로 wet-lab hit rate는 **base model + sampling + selection policy** 전체의 결과다.

---

## 6. Programmable design: generator를 proposal operator로 읽기

LDDM의 중요한 확장은 one-shot generation에서 끝나지 않는다는 점이다. Programmable design을 추상화하면

$$
x_{k+1}\sim q_\theta(\cdot\mid P,x_k,c_k),
$$

where $c_k$ contains fixed context or local constraints. Evaluator는

$$
u_k=U(x_k,P)
$$

를 계산하고 다음 design action에 사용한다.

이때 generator는 최종 optimizer 자체라기보다 **chemistry-aware proposal distribution**이다.

따라서 세 문제를 분리할 수 있다.

- generative prior가 좋은가?
- evaluator가 좋은 candidate를 구분하는가?
- search policy가 compute budget을 잘 배분하는가?

[[papers/sbdd/adaptiveflow|AdaptiveFlow]]가 ultra-large existing library에서 어떤 molecule을 expensive oracle에 보낼지 정한다면, LDDM은 local/generated chemical state에서 어떤 molecular modification을 제안할지를 learned generator로 수행한다고 볼 수 있다.

---

## 7. Synthesizable design: SA score가 아니라 reaction graph 위의 search

많은 generator는

$$
x\sim p_\theta(x\mid P)\rightarrow \operatorname{SA}(x)\text{ filter}
$$

를 사용한다. 그러나 낮은 SA score가 실제 available building blocks와 reaction route를 보장하지는 않는다.

LDDM public workflow는 building-block table, reaction SMARTS/templates, reaction-to-building-block role mapping을 입력으로 받는다.

$$
\mathcal X_{synth}=\{R(b_i,b_j):R\in\mathcal R,\ b_i,b_j\in\mathcal B\}.
$$

즉 synthetic feasibility가 generation 이후의 scalar penalty가 아니라 **generation/search domain** 자체에 들어간다.

### Reproducibility boundary

Official repo는 full prospective Enamine REAL reactions/building blocks를 license 때문에 배포하지 않는다. Public demo는 SynSpace-derived:

- 44,944 building blocks
- 3 reactions

을 제공한다.

따라서 public code로 workflow를 실행할 수 있다는 것과 paper의 full prospective chemical space를 그대로 재현할 수 있다는 것은 다르다.

---

## 8. Evidence를 네 층으로 분리하기

### Layer A — docking geometry

질문은 native-like pose를 생성할 수 있는가다. Symmetry-aware RMSD, chemical validity, clash/strain이 적합한 evidence다. 이것은 affinity를 직접 증명하지 않는다.

### Layer B — generative chemistry

Pocket-conditioned generation이 chemically valid하고 diverse하며 target-relevant한 candidates를 만드는가를 본다. Validity/diversity/novelty와 pocket geometry, synthesis constraint는 서로 다른 axis다.

### Layer C — prospective binding

bioRxiv abstract는 다섯 therapeutically relevant protein targets에서 designed/optimized ligands를 prospectively 검증했고, 모든 case에서 소수의 합성으로 confirmed binding을 확보했다고 저자들이 보고한다.

Retrospective docking benchmark보다 deployment-facing evidence가 강하지만 campaign별 generation count, filtering, synthesis budget, assay, hit definition이 결과 해석에 포함되어야 한다.

### Layer D — structural validation

Paper는 best designs 일부를 NMR spectroscopy와 X-ray crystallography로 characterize했다고 보고한다. Fraser Lab page는 38HB, 38HC, 38HY, 9SLI를 deposited structures로 연결하고, RCSB 38HB는 0.97 Å X-ray complex로 공개되어 있다.

이것은 selected success case의 binding-geometry evidence이지 전체 generated distribution의 unbiased pose-accuracy estimate는 아니다.

---

## 9. 실제 novelty

LDDM의 contribution은 세 층으로 나눌 수 있다.

### A. Unified conditional state completion

Docking, partial docking, fragment editing, de novo design을 같은 $(X,A,B)$ state와 masking/conditioning으로 표현한다.

### B. Mixed continuous–discrete generation

Coordinates는 flow dynamics로, atom/bond chemistry는 discrete Markov bridge로 함께 생성한다.

### C. Synthesizable programmable search

Generator를 one-shot outputter가 아니라 constrained proposal operator로 사용하고 reaction/building-block space를 search domain으로 넣는다.

Prospective experiments는 이 조합의 system-level utility를 보여주지만 각 component의 causal contribution을 따로 증명하지는 않는다.

---

## 10. Baseline fairness

Unified model은 task가 많아서 baseline comparison이 특히 어렵다.

### Docking

다음을 맞춰야 한다.

- receptor state
- pocket definition
- protonation/tautomer state
- reference-ligand information
- generated pose count
- integration budget
- ranking rule
- symmetry treatment

### Generation

다음을 맞춰야 한다.

- allowed molecule-size distribution
- scaffold/fragment constraints
- sample count
- property filters
- validity criteria
- synthetic-space constraints
- compute/oracle calls

### Prospective campaign

최종 hit rate는

$$
\text{hit rate}=f(\text{generator},\text{search},\text{filter},\text{selection},\text{assay},\text{synthesis budget})
$$

이므로 raw model score로 읽으면 안 된다.

---

## 11. Split and OOD boundary

Protein–ligand generative model의 generalization은 ligand split 하나로 설명할 수 없다.

| Axis | Leakage / interpolation risk |
| --- | --- |
| Ligand | close scaffold, analog series, stereoisomer |
| Protein | close sequence/structure family |
| Complex | near-identical pocket–ligand interaction pattern |
| Time | training cutoff 이전의 structures/assays |
| Pocket state | holo/apo/conformational-state shift |
| Chemistry | building blocks/reaction templates seen in training/search |
| Selection | test-target-specific evaluator tuning |

Docking과 design을 한 checkpoint에서 학습하면 같은 underlying complex가 다른 masking task로 노출될 수 있다. Split unit는 row가 아니라 **complex/family/scaffold identity**와 맞아야 한다.

Prospective wet-lab validation은 retrospective leakage 우려를 크게 줄이지만 broad OOD generalization을 자동으로 증명하지는 않는다. 다섯 selected targets에서의 success는 해당 campaign scope의 evidence다.

---

## 12. Reference-ligand boundary

Official examples는 protein과 함께 `--ref_ligand`를 전달한다. 이것은 pocket localization이나 coordinate reference를 위한 practical interface일 수 있다.

Benchmark에서는 다음을 분리해야 한다.

1. reference ligand가 pocket localization만 제공하는가?
2. exact pose/shape가 generative input에 들어가는가?
3. evaluated ligand 또는 close analog information이 들어가는가?
4. deployment에서도 같은 information이 available한가?

Known holo ligand를 이용한 site-defined generation과 apo/unliganded target discovery는 다른 task다. Reference-ligand availability를 숨긴 채 protein-only de novo design으로 일반화하면 안 된다.

---

## 13. Reproducibility

Public artifact surface는 강한 편이다.

- MIT source code
- executable docking/design examples
- Docker environment
- main paper checkpoint
- non-BindingNet checkpoint
- geometry reference
- public SynSpace-derived synthesis demo
- ODE sampling controls
- reaction-space preparation scripts

하지만 두 licensing boundary가 있다.

`CD+BB+BN`은 paper experiments에 사용된 checkpoint이며 BindingNet 때문에 CC BY-NC 4.0이다. Full Enamine REAL reaction/building-block space는 별도 license가 필요하고 repository에서 재배포되지 않는다.

최소 reproduction contract는 다음과 같다.

```text
paper version
+ repository commit
+ checkpoint identity/license
+ pocket preparation
+ reference-ligand role
+ sampler / steps / noise
+ number of samples
+ candidate selection rule
+ chemical-space release
```

Public artifacts가 충분하다는 것과 full prospective campaign을 byte-for-byte 재현할 수 있다는 것은 다르다.

---

## 14. 주요 confounders

### Unified-model gain vs data gain

Multi-task model은 heterogeneous supervision을 더 많이 사용한다. Gain이 masking/unification 때문인지 data volume/coverage 때문인지 matched single-task control이 필요하다.

### Generator vs evaluator/search

Prospective candidate는 base generator output을 무작위로 합성한 것이 아니다. Search와 evaluator/filter가 개입한다. Wet-lab result는 base likelihood의 직접 측정값이 아니다.

### Synthesizability vs practical synthesis

Reaction-template reachable은 SA score보다 실제 chemistry에 가깝지만 yield, conditions, protecting groups, vendor availability, cost, purification을 모두 보장하지 않는다.

### Structural-confirmation selection bias

NMR/X-ray 성공 사례는 매우 가치 있지만 selected hits에 집중될 수 있다. Mechanistic pose evidence이지 generated set 전체의 calibration curve는 아니다.

### Preprint status

2026-09-18 공개 bioRxiv preprint이므로 peer-review 과정에서 method/result wording이 수정될 수 있다. Version/date pinning이 필요하다.

---

## 15. Decision-useful ablations

### A. Unified vs task-specific

동일 examples와 total compute에서

```text
A0: docking-only
A1: design-only
A2: unified masked model
```

을 비교한다. Docking supervision이 design에 positive transfer를 주는지, design supervision이 pose quality를 개선하는지, task interference가 어디에서 생기는지 본다.

### B. Partial-state curriculum

```text
B0: full docking + full design only
B1: + fragment growing/linking masks
B2: + local/random partial-coordinate masks
```

로 partial completion training이 genuinely reusable representation을 만드는지 본다.

### C. Synthesis-aware search vs generate-then-filter

Matched total generator/scoring calls에서

```text
C0: unconstrained generation → SA/retro filter
C1: reaction-constrained generation/search
```

를 비교한다. Final quality, diversity, route validity, building-block coverage, oracle calls per accepted candidate를 같이 봐야 한다.

### D. Reference-ligand dependence

```text
D0: known holo reference ligand
D1: pocket center only
D2: apo structure / predicted pocket
```

를 비교하면 deployment boundary가 드러난다.

### E. Sampling-budget curve

$K\in\{1,5,10,50,100\}$에서 top-ranked pose, oracle pose, diversity, validity, compute를 기록해 generator capacity와 selector quality를 분리한다.

---

## 16. Falsification

다음 결과가 나오면 one-model-for-many-SBDD-tasks story의 mechanistic advantage는 약해진다.

1. 같은 data/compute에서 task-specific models가 주요 task에서 일관되게 우세하다.
2. Unified gain이 단순 training-set enlargement control에서 사라진다.
3. Partial masks를 제거해도 fragment/docking transfer가 변하지 않는다.
4. Reaction-constrained search가 matched compute에서 generate-then-filter보다 quality–diversity–synthesizability Pareto를 개선하지 못한다.
5. Reference ligand를 제거하면 성능이 급락하고 pocket-only setup에서 회복되지 않는다.
6. Prospective success가 다른 protein-family/assay context에서 재현되지 않는다.

반대로 이런 controls를 통과한다면 특정 leaderboard score보다 **partial-state molecular generation이라는 reusable SBDD interface**가 LDDM의 더 강한 contribution이 된다.

---

## 17. 실무 연구 방향: unified proposal model + independent verifier

LDDM이 던지는 좋은 질문은 하나의 giant generator가 모든 것을 해야 하는가가 아니다.

> **Dock → refine → grow → redesign을 서로 다른 representation으로 넘기지 않고, 하나의 structured molecular state에서 uncertainty와 constraints를 유지한 채 연속적으로 수행할 수 있는가?**

Generic SBDD pipeline은 다음처럼 생각할 수 있다.

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

Independent scoring을 남겨 두는 것이 중요하다. Generator와 같은 error mode를 가진 evaluator만 사용하면 correlated error가 iterative selection에서 증폭될 수 있다.

따라서 practical baseline은 **unified proposal model + independent verifier**가 적절하다.

---

## 18. Pose generation과 affinity prediction을 섞지 말 것

LDDM의 주요 contribution은 generative structure/design 쪽이다.

Native-like pose를 잘 만든다는 것은

$$
p(X\mid A,B,P)
$$

를 잘 모델링한다는 evidence다.

Binding affinity는 $\Delta G_{bind}$ 또는 assay-specific potency라는 다른 target이다. Pose가 좋아도 desolvation, entropy, protonation, water network, receptor reorganization, assay context를 충분히 모델링하지 않으면 affinity ranking은 틀릴 수 있다.

Prospective binder discovery가 성공했다고 해서 generator가 calibrated affinity predictor라는 결론은 나오지 않는다. 이 구분은 [[concepts/sbdd/pose-quality|Pose quality]]와 [[concepts/sbdd/binding-affinity|Binding affinity]]를 같이 읽어야 하는 이유다.

---

## 19. Prospective evidence를 읽는 campaign card

Prospective experiment는 retrospective split보다 leakage-resistant하다. 하지만 prospective라는 단어만으로 comparison fairness가 자동 보장되지는 않는다.

Decision-useful campaign card에는 적어도 다음이 필요하다.

- target novelty relative to training
- starting ligand/fragment availability
- generated count
- filtered/scored count
- human selection 여부
- synthesized count
- assay endpoint and threshold
- confirmed hit count
- structural follow-up selection rule

BioRxiv abstract는 `five targets`, `small number synthesized`, `confirmed binding`, `NMR/X-ray on best designs`라는 강한 high-level evidence를 제공한다. Target별 exact budget과 affinity distribution을 abstract만으로 완전히 복원할 수는 없다.

따라서 **prospective evidence가 존재한다**는 것과 **모든 campaign efficiency가 독립 재현되었다**는 것을 분리해야 한다.

---

## 20. Related notes

- [[papers/sbdd/posebusters|PoseBusters]] — generated/docked pose의 chemical/geometric validity boundary
- [[papers/sbdd/adaptiveflow|AdaptiveFlow]] — expensive oracle 아래 candidate-selection policy를 최적화하는 complementary view
- [[papers/generative-models/ensemble-conditioned-molecular-design|Ensemble-Conditioned Molecular Design]] — inference-time condition composition으로 multi-state target/avoid objective를 만드는 다른 route
- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]

LDDM과 AdaptiveFlow를 같이 보면 두 search problem을 분리할 수 있다.

$$
\text{AdaptiveFlow}:\ \text{which existing molecules should be evaluated?}
$$

$$
\text{LDDM}:\ \text{which new molecular state/change should be proposed?}
$$

둘 다 architecture만큼 search/allocation policy가 중요하다.

---

## Final verdict

LDDM의 headline은 3D molecule generator가 prospective hits를 냈다는 것이지만 오래 남는 contribution은 더 구조적이다.

첫째, docking과 de novo design 사이의 경계를 없애고 **partial molecular state completion**이라는 하나의 interface로 다시 썼다. Molecular graph와 coordinates를 joint state로 두면 pose-only, partial-coordinate, fragment, full-generation task를 mask만 바꿔 표현할 수 있다.

둘째, synthesizability를 scalar filter로만 보지 않고 **reaction/building-block-defined action space**로 바꿨다. 이것은 생성 quality뿐 아니라 실제 search efficiency와 실험 가능성을 architecture 주변의 first-class objective로 만든다.

셋째, prospective binding과 public structural artifacts가 있어 benchmark-only generative paper보다 evidence가 강하다. 동시에 full Enamine REAL space licensing, main checkpoint의 non-commercial license, selection/search confound, target별 experimental-budget detail 같은 reproducibility boundary가 남아 있다.

따라서 이 paper를 인용할 때 가장 안전한 문장은 다음과 같다.

> **LDDM은 하나의 pocket-conditioned mixed continuous–discrete generator를 masking/conditioning으로 재사용해 docking부터 molecular design까지 여러 SBDD task를 통합하고, reaction-space-constrained search와 prospective experiments를 통해 이 unified formulation의 practical potential을 보여준다.**

`모든 SBDD task에서 최적이다`, `affinity를 정확히 예측한다`, `full prospective campaign이 완전히 공개 재현 가능하다`는 더 강한 주장은 현재 evidence와 분리해야 한다.

---

## Three durable takeaways

1. **Docking과 molecular generation은 partial-state completion의 서로 다른 mask로 통일할 수 있다.** Graph, bond, coordinate를 joint state로 두면 dock/grow/link/de novo를 같은 model surface에서 표현할 수 있고, 실제 연구 질문은 task sharing이 positive transfer를 만드는지 matched ablation으로 확인하는 것이다.

2. **Synthesizability는 post-hoc score보다 action-space constraint로 넣을 때 더 강한 의미를 갖는다.** Reaction templates와 available building blocks가 generation/search domain을 정의하면 “만들 수 있을 법한 molecule”이 아니라 “주어진 chemical space에서 구성 가능한 molecule” 쪽으로 objective가 이동한다.

3. **Prospective evidence도 generator, selector, chemistry space, assay를 분리해서 읽어야 한다.** Confirmed binders와 NMR/X-ray structures는 중요한 evidence지만, end-to-end campaign success는 base model 하나가 아니라 sampling, programmable search, filtering, synthesis selection, assay의 합성 결과다.

---

## Sources

- Igashov I, Schneuing A, Dobbelstein AW, et al. [A Unified 3D Generative Model for Synthesizable Structure-Based Drug Design](https://doi.org/10.64898/2026.09.15.751537). bioRxiv, posted 2026-09-18.
- Official implementation: [LPDI-EPFL/lddm](https://github.com/LPDI-EPFL/lddm), inspected at commit [`f254fb4f8525b3803e79eb95e9f1a163fe8b2459`](https://github.com/LPDI-EPFL/lddm/commit/f254fb4f8525b3803e79eb95e9f1a163fe8b2459).
- Public checkpoints and reference data: [Zenodo record 22754501](https://zenodo.org/records/22754501).
- Independent experimental structure record: [RCSB PDB 38HB](https://www.rcsb.org/structure/38HB).
- Paper-linked structure list: [Fraser Lab publications](https://fraserlab.com/publications/).
