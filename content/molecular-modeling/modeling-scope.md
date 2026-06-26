---
title: Computational Biology Scope
aliases:
  - bio/modeling-scope
tags:
  - molecular-modeling
---

# Computational Biology Scope

Use `Computational Biology` as the public umbrella for molecule, ligand, protein, pocket, complex, conformer, docking, structure-based modeling, and sequence-level genome objects. Use `Molecular Modeling` for the molecule/structure/docking-heavy subset. Use `AI` for the method layer: architectures, learning signals, objectives, generative models, evaluation, and systems.

The same paper can touch both, but the page title should follow the strongest claim:

$$
\text{route}
=
\operatorname*{arg\,max}_{a \in \mathcal{A}}
\operatorname{centrality}(a)
$$

where $\mathcal{A}$ can include object, representation, architecture, objective, benchmark, system, or agent workflow.

## Naming Rule

| Center | Prefer | Avoid |
| --- | --- | --- |
| Protein, ligand, molecule, pocket, complex | Computational Biology / Molecular Modeling | Overbroad bio label |
| Docking, pose, conformer, virtual screening | Structure-Based Modeling | Calling the task itself AI |
| GNN, Transformer, diffusion, flow matching, SSL | AI | Hiding the method under Molecular Modeling |
| Symmetry, coordinate frame, likelihood, loss | Math | Treating equations as paper-specific details |
| Split, metric, leakage, benchmark protocol | Evaluation | Reporting only headline scores |
| Tool-using LLM workflow | Agents | Mixing agent ops into model architecture notes |

## Practical Routes

| Question | Route |
| --- | --- |
| What is being modeled? | [Entities](/molecular-modeling/entities), [Molecules](/molecular-modeling/molecules), [Proteins](/molecular-modeling/proteins) |
| Is it docking, pose, conformer, or screening? | [Structure-based modeling](/molecular-modeling/structure-based), [Docking](/molecular-modeling/docking) |
| What model family is used? | [AI architectures](/ai/architectures), [Learning methods](/ai/learning-methods) |
| What equation explains the claim? | [Math](/math), [Formula intake](/math/formula-intake) |
| What evidence supports the claim? | [Data and evaluation](/molecular-modeling/data-evaluation), [Evaluation](/ai/evaluation) |
| Is the note ready to publish? | [AI Computational Biology Math readiness gate](/papers/workflows/ai-molecular-math-readiness-gate) |

## Examples

| Topic | Primary Home | Secondary Links |
| --- | --- | --- |
| Protein-ligand docking | [Docking](/molecular-modeling/docking) | [Scoring function](/concepts/sbdd/scoring-function), [Pose RMSD](/concepts/sbdd/pose-rmsd) |
| Ligand conformer generation | [Molecules](/molecular-modeling/molecules) | [Conformer](/concepts/molecular-modeling/conformer), [Coordinate modeling contract](/concepts/geometric-deep-learning/coordinate-modeling-contract) |
| Equivariant pose prediction | [Structure-based modeling](/molecular-modeling/structure-based) | [Equivariance](/concepts/geometric-deep-learning/equivariance), [Coordinate prediction](/concepts/tasks/coordinate-prediction) |
| Molecular graph pretraining | [Molecules](/molecular-modeling/molecules) | [Self-supervised learning](/ai/learning-methods), [Graph neural networks](/concepts/architectures/gnn) |
| Flow-matching molecule generation | [Molecules](/molecular-modeling/molecules) | [Flow matching](/concepts/generative-models/flow-matching), [Generative models](/ai/generative-models) |

## Checks

- Does the title name the object or workflow before the model brand?
- Does the note separate domain claims from AI method claims?
- Are conformers, poses, and protein-bound geometries distinguished?
- Are formulas linked to Math notes instead of left as unexplained symbols?
- Are benchmark claims tied to split, metric, baseline, and leakage risk?

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/computational-biology|Computational Biology Boundary]]
- [[molecular-modeling/structure-based/index|Structure-Based Modeling]]
- [[molecular-modeling/docking|Docking]]
- [[concepts/coverage-matrix|Coverage Matrix]]
