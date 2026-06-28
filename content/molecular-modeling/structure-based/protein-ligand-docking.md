---
title: Protein-Ligand Docking
aliases:
  - computational-biology/structure-based/protein-ligand-docking
  - research/structure-based-ai/protein-ligand-docking
tags:
  - docking
  - structure-based-drug-discovery
  - protein-ligand
---

# Protein-Ligand Docking

Protein-ligand docking estimates how a small molecule may bind to a protein binding site. A useful docking workflow separates pose generation, pose filtering, and affinity or ranking models.

Docking can be framed as searching over ligand pose variables:

$$
X^\* = \arg\min_X S(P, L, X)
$$

Here $P$ is the protein or [[entities/pocket|pocket]], $L$ is the ligand, $X$ is the ligand pose/conformation in the binding site, and $S$ is a scoring function. In learned methods, $S$ may be replaced or complemented by a generative model.

More explicitly, the ligand pose can be written as:

$$
q=(t,R,\tau),
\qquad
X = X(q)
$$

where $t$ is translation, $R$ is rotation, and $\tau$ is a vector of torsion angles. Classical docking searches over $q$; learned pose generators may sample $q$ or directly predict $X$.

## What Docking Produces

Docking output is usually a set of candidate ligand poses in a binding site. A pose is not automatically a binding explanation. It has to be checked for geometry, chemistry, protein-ligand contacts, and whether the assumed binding site is meaningful.

## Modeling Views

- Search problem: explore ligand translations, rotations, conformers, and sometimes side-chain flexibility.
- Pose generation problem: sample or optimize candidate geometries conditioned on protein and ligand context.
- Ranking problem: use a [[concepts/sbdd/scoring-function|scoring function]] to order candidate poses.
- Generative problem: directly sample plausible complex structures with learned geometry.
- Evaluation problem: distinguish native-like poses from invalid or overfit predictions.
- Screening problem: rank a ligand library for [[concepts/sbdd/virtual-screening|virtual screening]].

## Working Questions

- How should pose plausibility be checked before using a [[concepts/sbdd/scoring-function|scoring function]]?
- Which failures are geometric, chemical, or data-driven?
- How should docking outputs be evaluated against physical constraints and benchmarks such as [[papers/sbdd/posebusters|PoseBusters]]?
- How much protein flexibility is needed for the target being modeled?
- Is the method using a known pocket, a predicted pocket, or blind docking?

## Pipeline Sketch

1. Prepare receptor and ligand structures.
2. Generate candidate poses with an explicit [[concepts/sbdd/pose-generation|pose generation]] policy.
3. Filter implausible poses using geometry and chemistry checks.
4. Rank candidates with a [[concepts/sbdd/scoring-function|scoring function]].
5. Record uncertainty and failure modes.

## Component Equations

A useful decomposition is:

$$
\{\hat{X}_k\}_{k=1}^{K}
\sim
G_\theta(P,L,c)
$$

$$
s_k
=
S_\phi(P,L,\hat{X}_k)
$$

$$
\hat{X}_{\mathrm{top}}
=
\operatorname*{select}_{k}
(\hat{X}_k,s_k)
$$

This separates generator $G_\theta$, scoring function $S_\phi$, and selection rule. If a paper changes all three, the improvement cannot be assigned to only one component.

## Failure Modes

- Plausible-looking pose with invalid ligand geometry.
- Low score caused by scoring shortcuts rather than real binding signal.
- Correct pocket but wrong ligand orientation.
- Good RMSD on an easy split but poor generalization to new proteins or scaffolds.
- Protein treated as rigid when the relevant binding mode requires flexibility.
- Score convention inversion between docking tool, learned scorer, and benchmark script.
- Filtering invalid poses before reporting metrics without counting filtered failures.

## Practical Checks

- Validate ligand identity, stereochemistry, bond lengths, clashes, and pocket overlap.
- Report [[concepts/sbdd/pose-quality|pose quality]] separately from [[concepts/sbdd/binding-affinity|binding affinity]] or enrichment metrics.
- Compare against classical docking baselines when possible.
- Track whether evaluation uses crystal structures, predicted structures, apo structures, or sequence-only input.
- Use [[papers/sbdd/posebusters|PoseBusters]]-style plausibility checks before claiming state-of-the-art structure generation.

## Related

- [[entities/pocket|Pocket]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[concepts/evaluation/leakage|Leakage]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
