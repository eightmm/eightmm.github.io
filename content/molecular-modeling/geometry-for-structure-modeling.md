---
title: Geometry for Structure Modeling
aliases:
  - computational-biology/geometry-for-structure-modeling
  - comp-bio/geometry-for-structure-modeling
tags:
  - computational-biology
  - geometry
  - structure-based-modeling
---

# Geometry for Structure Modeling

Structure modeling에서 geometry는 그림이 아니라 coordinate, distance, frame, symmetry, constraint를 다루는 언어입니다. Protein, ligand, pocket, complex를 다룰 때는 어떤 값이 좌표계에 의존하고 어떤 값이 보존되어야 하는지 먼저 정해야 합니다.

$$
X \in \mathbb{R}^{n \times 3},
\qquad
X' = RX + t,
\qquad
R \in SO(3),\; t \in \mathbb{R}^3
$$

순수 수학으로서의 symmetry와 group action은 [[math/geometry-symmetry|Geometry and Symmetry]]에 둡니다. 여기서는 그 언어를 protein, ligand, pocket, complex의 structure modeling claim에 적용합니다.

## Geometry Contract

Structure modeling note는 geometry를 쓰기 전에 아래 contract를 고정해야 합니다.

$$
\mathcal{C}_{\mathrm{geom}}
=
(\mathcal{U},\ X,\ A,\ F,\ G,\ V,\ M)
$$

| Part | Meaning | Example |
| --- | --- | --- |
| $\mathcal{U}$ | biological unit | atom, residue, ligand, pocket, chain, complex |
| $X$ | coordinate tensor | atom coordinates, residue C-alpha coordinates, ligand conformer |
| $A$ | atom or residue identity mapping | atom correspondence, residue index, chain ID |
| $F$ | coordinate frame | global frame, pocket-aligned frame, local residue frame |
| $G$ | allowed symmetry group | rotation, translation, permutation, ligand symmetry |
| $V$ | validity constraints | bond length, chirality, clash, pocket availability |
| $M$ | metric and alignment rule | RMSD, lDDT, clash count, strain, interaction geometry |

Without this contract, the same number can support different claims. For example, an RMSD value means different things under ligand-only alignment, pocket alignment, global protein alignment, or symmetry-aware atom matching.

## Core Objects

| Object | Meaning | Route |
| --- | --- | --- |
| Coordinate matrix | atom/residue/point 좌표 $X$ | [Coordinate frame](/concepts/geometric-deep-learning/coordinate-frame) |
| Distance | pairwise geometry invariant | [Distance geometry](/concepts/geometric-deep-learning/distance-geometry) |
| Frame | local coordinate basis or global reference | [Coordinate frame](/concepts/geometric-deep-learning/coordinate-frame) |
| Group action | rotation, translation, permutation이 object에 작용하는 방식 | [Symmetry group](/concepts/math/symmetry-group) |
| Invariance | input이 변해도 output scalar가 보존됨 | [Equivariance](/concepts/geometric-deep-learning/equivariance) |
| Equivariance | input transform에 맞춰 output도 transform됨 | [Equivariance](/concepts/geometric-deep-learning/equivariance) |

## Coordinate Source

Coordinate source changes the meaning of the learning target and the evaluation claim.

| Source | Useful for | Main risk |
| --- | --- | --- |
| experimental structure | reference geometry, known complex, benchmark target | resolution, missing atoms, alternate conformations |
| predicted protein structure | scalable target representation | prediction error becomes hidden input noise |
| docked pose | generated candidate geometry | docking bias or circular evaluation |
| minimized pose | physically relaxed geometry | force-field assumptions alter the target |
| generated coordinates | model output | validity and constraint repair must be counted |
| ligand-defined pocket | local binding-site context | test ligand leakage if unavailable at inference |

When coordinates come from different sources, do not treat them as the same label unless the note explains the transformation and filtering policy.

## Quantity Type

| Quantity | Transform behavior | Example |
| --- | --- | --- |
| Scalar invariant | $s(RX+t)=s(X)$ | affinity, energy, RMSD after alignment |
| Vector equivariant | $v(RX+t)=Rv(X)$ | force, displacement |
| Coordinate equivariant | $Y(RX+t)=RY(X)+t$ | generated pose, denoised coordinates |
| Pair distance | unchanged by rigid motion | contact, distance matrix |
| Local frame | rotates with structure | residue frame, ligand frame |

The required model behavior follows the target type:

$$
\begin{aligned}
\text{scalar target:} && f(RX+t) &= f(X) \\
\text{vector target:} && v(RX+t) &= Rv(X) \\
\text{coordinate target:} && Y(RX+t) &= RY(X)+t
\end{aligned}
$$

## Structure Claim Map

| Claim | Required geometry check |
| --- | --- |
| Pose prediction | coordinate frame, atom mapping, RMSD alignment, clash, strain |
| Affinity scoring | pose quality와 score meaning을 분리 |
| Structure generation | generated coordinates의 distance, bond, chirality, clash constraint |
| Equivariant model | scalar output은 invariant, coordinate/vector output은 equivariant |
| Pocket modeling | pocket definition이 inference time에 사용 가능한 정보만 쓰는지 확인 |

## Metric and Alignment

Geometry metrics are not interchangeable. The alignment rule is part of the metric.

| Metric | Good for | Check |
| --- | --- | --- |
| ligand RMSD | pose accuracy against a reference ligand | atom mapping, symmetry, alignment rule |
| pocket RMSD | local receptor movement | residue selection and missing atoms |
| distance error | invariant geometry comparison | loss of orientation or chirality information |
| clash count | physical plausibility | atom radii, hydrogens, protonation state |
| strain or torsion quality | ligand conformer plausibility | force field and tautomer/protonation policy |
| interaction pattern | contact or pharmacophore consistency | distance thresholds and pocket definition |

For a generated set, report the denominator:

$$
\text{valid pose rate}
=
\frac{\#\text{poses passing geometry checks}}
{\#\text{generated poses before filtering}}
$$

## Common Equations

Distance matrix:

$$
D_{ij} = \lVert x_i - x_j \rVert_2
$$

Invariant scalar:

$$
f(RX+t)=f(X)
$$

Equivariant coordinate output:

$$
g(RX+t)=Rg(X)+t
$$

Force-like vector:

$$
F_i(RX+t)=R F_i(X)
$$

Coordinate update:

$$
X^{(k+1)} = X^{(k)} + \Delta X_\theta(H^{(k)}, X^{(k)}, E)
$$

If $\Delta X_\theta$ is coordinate displacement, it must transform equivariantly under rotation. If the model predicts only a scalar score, the score should be invariant to global rigid motion.

## Leakage Checks

| Check | Why |
| --- | --- |
| ligand-defined pocket | test ligand pose가 pocket extraction에 들어가면 deployment보다 쉬워짐 |
| template structure | homolog/template이 split을 넘어가면 structure task가 과대평가됨 |
| coordinate source | predicted, experimental, docked, minimized coordinate를 섞으면 label meaning이 바뀜 |
| atom mapping | generated ligand와 reference ligand의 atom correspondence가 불명확하면 RMSD가 흔들림 |
| alignment rule | global alignment, pocket alignment, ligand-only alignment가 다른 claim을 만듦 |

## Where This Fits

| If the note is about | Put it in |
| --- | --- |
| group action, metric space, symmetry definition | [[math/geometry-symmetry|Geometry and Symmetry]] |
| equivariant layers or coordinate message passing | [[concepts/geometric-deep-learning/index|Geometric deep learning]] |
| protein, ligand, pocket, complex geometry | this page or [[molecular-modeling/structure-based/index|Structure-Based Modeling]] |
| docking pose plausibility | [[concepts/sbdd/pose-quality|Pose quality]] |
| model objective and sampler | [[ai/generative-models|Generative Models]] |

## Related

- [[math/geometry-symmetry|Geometry and Symmetry]]
- [[molecular-modeling/structure-based/index|Structure-Based Modeling]]
- [[molecular-modeling/geometry|Computational Biology Geometry Route]]
- [[concepts/sbdd/pocket-definition-contract|Pocket definition contract]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
