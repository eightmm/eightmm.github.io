---
title: Coverage Matrix
tags:
  - concepts
  - ai
  - molecular-modeling
  - math
  - papers
---

# Coverage Matrix

Coverage matrix is a routing sheet for new papers, posts, and concept notes. It helps decide which reusable notes should exist before a topic becomes a public post or curated paper note.

$$
\text{coverage}
=
\text{object}
+ \text{representation}
+ \text{task}
+ \text{model}
+ \text{objective}
+ \text{evidence}
+ \text{public boundary}
$$

## Intake Routes

| Need | Use | Start |
| --- | --- | --- |
| Reusable wiki note | definition, boundary, contract, checks, links, public boundary | [Wiki note quality gate](/concepts/wiki-note-quality-gate) |
| AI paper claim | architecture, objective, evidence, and system boundary | [AI paper intake](/ai/paper-intake) |
| Molecular modeling paper claim | object, representation, label, split, leakage | [Molecular modeling paper intake](/molecular-modeling/paper-intake) |
| Formula or objective | symbols, distributions, derivatives, metrics | [Formula intake](/math/formula-intake) |
| Benchmark score | data, task, split, metric, allowed information, reporting | [Benchmark intake](/concepts/data/benchmark-intake) |
| Korean synthesis post | reader question, main axis, minimum formulas, wiki links | [AI-Molecular-Math post intake](/posts/ai-molecular-math-post-intake) |
| Topic map | reader question, primary axis, object boundary, methods, evidence, next path | [Topic map contract](/concepts/topic-map-contract) |
| Paper review workflow | metadata, claims, evidence, artifacts, reproduction | [Paper review workflow](/papers/workflows/paper-review-workflow) |
| Claim route | primary axis, secondary axes, bucket, concept updates | [Claim routing](/papers/workflows/claim-routing) |
| Concept update | reusable definitions, formulas, contracts, evidence boundaries | [Concept update contract](/papers/workflows/concept-update-contract) |
| Promotion readiness | route, representation, objective, evidence, and public boundary | [AI-Molecular-Math readiness gate](/papers/workflows/ai-molecular-math-readiness-gate) |
| Fillable paper note | reusable skeleton for AI/molecular modeling/Math papers | [AI-Molecular-Math paper template](/papers/workflows/ai-molecular-math-paper-template) |
| Paper bucket | decide whether the paper belongs in SBDD, protein modeling, architecture, generation, learning methods, or systems | [Paper triage](/papers/workflows/paper-triage) |

## Axis Coverage

| Axis | Question | Canonical Area |
| --- | --- | --- |
| Object | What entity is being modeled? | [Entities](/entities), [Molecular modeling entities](/molecular-modeling/entities) |
| Modality | What form is the input or output? | [Modalities](/concepts/modalities) |
| Representation | How does the raw object become tokens, graph, coordinates, embedding, or features? | [Representation contract](/concepts/modalities/representation-contract), [Tensor shape notation](/concepts/math/tensor-shape-notation) |
| Task | What output space and validity rule define success? | [Tasks](/concepts/tasks) |
| Coordinate contract | What frame, symmetry, mapping, loss, and metric define coordinate claims? | [Coordinate modeling contract](/concepts/geometric-deep-learning/coordinate-modeling-contract) |
| Data | What example, label, split, and preprocessing contract define the dataset? | [Data](/concepts/data) |
| Architecture | What inductive bias and complexity does the model use? | [Architectures](/concepts/architectures), [AI architectures](/ai/architectures) |
| Learning method | What supervision, pretraining, transfer, or preference signal is used? | [Learning methods](/concepts/learning), [AI learning methods](/ai/learning-methods) |
| Objective-metric link | Does the optimized loss support the reported metric and utility claim? | [Objective-metric alignment](/concepts/machine-learning/objective-metric-alignment) |
| Generative model | What distribution, score, velocity, or sampling path is modeled? | [Generative models](/concepts/generative-models), [AI generative models](/ai/generative-models) |
| Math | Which formula, object type, distribution, or estimator is needed? | [Math](/math), [Math foundations](/concepts/math) |
| Evaluation | Which metric, baseline, split, uncertainty, and failure mode support the claim? | [Evaluation](/ai/evaluation), [Evaluation concepts](/concepts/evaluation) |
| Claim boundary | What exactly does the evidence prove, and where does it stop? | [Claim-evidence boundary](/concepts/evaluation/claim-evidence-boundary) |
| Systems | Does the claim depend on training, serving, tools, reproducibility, or compute? | [AI systems](/concepts/systems), [Infra](/infra) |
| Agent workflow | Does an LLM use tools, memory, planning, or verification? | [Agents](/agents) |

## Molecular Modeling Coverage

| Topic | Must Cover | Route |
| --- | --- | --- |
| Molecule or ligand model | standardization, identity, representation, scaffold split, property or interaction label | [Molecules](/molecular-modeling/molecules) |
| Protein model | sequence, structure, domains, residue indexing, family split, representation | [Proteins](/molecular-modeling/proteins) |
| Protein-ligand complex | protein context, ligand context, pocket, pose, interaction, split on both axes | [Structure-based modeling](/molecular-modeling/structure-based) |
| Docking or virtual screening | receptor/ligand preparation, pose generation, scoring, filtering, enrichment metric | [Docking](/molecular-modeling/docking) |
| Geometry-heavy model | coordinate frame, invariant/equivariant target, rotation/translation behavior, leakage risk | [Molecular Modeling geometry](/molecular-modeling/geometry) |
| Genome sequence model | region, k-mer, variant, sequence context, split and annotation boundary | [Genome](/molecular-modeling/genome) |
| Molecular Modeling benchmark | label semantics, assay/source, split unit, leakage, baseline, metric | [Data and evaluation](/molecular-modeling/data-evaluation) |

## Minimum Note Bundle

Before promoting a paper cluster into a Korean post, check that the bundle has:

- One route page: [[ai/paper-intake|AI paper intake]], [[molecular-modeling/paper-intake|Molecular modeling paper intake]], or [[math/formula-intake|Formula intake]].
- One wiki-note quality pass before a concept becomes a reusable anchor: [[concepts/wiki-note-quality-gate|Wiki note quality gate]].
- One data or benchmark page when a score is central: [[concepts/data/benchmark-intake|Benchmark intake]].
- One topic-map contract when a cluster spans several axes: [[concepts/topic-map-contract|Topic map contract]].
- One object/modality/task page when the input or output is nontrivial.
- One representation contract when preprocessing or featurization changes the object seen by the model.
- One tensor-shape pass when a formula depends on batch, token, graph, coordinate, head, or candidate axes.
- One coordinate modeling contract when a paper predicts poses, conformers, structures, vectors, forces, or coordinate updates.
- One architecture or learning-method page when the method is central.
- One objective-metric alignment check when the training loss and reported metric differ.
- One evaluation page explaining metric, split, baseline, uncertainty, or failure mode.
- One claim boundary when the result could be overread: [[concepts/evaluation/claim-evidence-boundary|Claim-evidence boundary]].
- One paper note or paper-analysis note when a specific paper claim is being discussed.
- One claim-routing pass when a paper or post candidate spans multiple axes: [[papers/workflows/claim-routing|Claim routing]].
- One concept-update pass when a paper changes reusable definitions, formulas, contracts, or evidence boundaries: [[papers/workflows/concept-update-contract|Concept update contract]].
- One readiness gate pass before promoting a multi-axis candidate: [[papers/workflows/ai-molecular-math-readiness-gate|AI-Molecular-Math readiness gate]].
- One fillable template pass for mixed AI/molecular modeling/Math papers: [[papers/workflows/ai-molecular-math-paper-template|AI-Molecular-Math paper template]].
- One Korean post intake pass when the result is reader-facing: [[posts/ai-molecular-math-post-intake|AI-Molecular-Math post intake]].

## Checks

- Is the central axis clear: object, method, formula, benchmark, paper cluster, or project?
- Are AI, molecular modeling, and Math links separated instead of collapsed into one vague topic?
- Does every performance claim have a route to benchmark intake or evaluation protocol?
- Does every molecular modeling claim name object, label, split, and leakage risk?
- Does every formula-heavy claim define symbols and the sampled distribution?
- Is the public boundary clear before adding a post or paper note?

## Related

- [[concepts/index|Concepts]]
- [[concepts/wiki-note-quality-gate|Wiki note quality gate]]
- [[ai/index|AI]]
- [[molecular-modeling/index|Molecular Modeling]]
- [[math/index|Math]]
- [[papers/index|Papers]]
- [[concepts/topic-map-contract|Topic map contract]]
- [[posts/ai-molecular-math-post-intake|AI-Molecular-Math post intake]]
- [[papers/workflows/ai-molecular-math-readiness-gate|AI-Molecular-Math readiness gate]]
- [[papers/workflows/concept-update-contract|Concept update contract]]
