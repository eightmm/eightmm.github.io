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
| AI claim pattern | architecture, learning, generation, evaluation, scaling, systems, or agent workflow | [AI paper claim patterns](/ai/paper-claim-patterns) |
| Computational biology paper claim | object, representation, label, split, leakage | [Computational Biology paper intake](/molecular-modeling/paper-intake) |
| Computational biology claim pattern | property, activity, docking, generation, protein design, genome sequence modeling | [Computational Biology paper claim patterns](/molecular-modeling/paper-claim-patterns) |
| Formula or objective | symbols, distributions, derivatives, metrics, constraints | [Formula intake](/math/formula-intake) |
| Formula pattern | common objective or metric pattern before detailed derivation | [Formula pattern catalog](/math/formula-patterns) |
| AI + computational biology + Math claim | object, representation, model, objective, evidence, public boundary | [AI Computational Biology Math contract](/concepts/ai-computational-biology-math-contract) |
| Benchmark score | data, task, split, metric, allowed information, reporting | [Benchmark intake](/concepts/data/benchmark-intake) |
| Korean synthesis post | reader question, main axis, minimum formulas, wiki links | [AI Computational Biology Math post intake](/posts/ai-molecular-math-post-intake), [Wiki bundle checklist](/posts/wiki-bundle-checklist) |
| Topic map | reader question, primary axis, object boundary, methods, evidence, next path | [Topic map contract](/concepts/topic-map-contract) |
| Paper review workflow | metadata, claims, evidence, artifacts, reproduction | [Paper review workflow](/papers/workflows/paper-review-workflow) |
| Claim route | primary axis, secondary axes, bucket, concept updates | [Claim routing](/papers/workflows/claim-routing) |
| Paper to wiki extraction | claim, object, representation, method, formula, data, evidence, artifact updates | [Paper to wiki extraction](/papers/workflows/paper-to-wiki-extraction) |
| Concept update | reusable definitions, formulas, contracts, evidence boundaries | [Concept update contract](/papers/workflows/concept-update-contract) |
| Promotion readiness | route, representation, objective, evidence, and public boundary | [AI Computational Biology Math readiness gate](/papers/workflows/ai-molecular-math-readiness-gate) |
| Fillable paper note | reusable skeleton for AI/computational biology/Math papers | [AI Computational Biology Math paper template](/papers/workflows/ai-molecular-math-paper-template) |
| Paper bucket | decide whether the paper belongs in SBDD, protein modeling, architecture, generation, learning methods, or systems | [Paper triage](/papers/workflows/paper-triage) |

## Axis Coverage

| Axis | Question | Canonical Area |
| --- | --- | --- |
| Object | What entity is being modeled? | [Entities](/entities), [Computational biology entities](/molecular-modeling/entities) |
| Modality | What form is the input or output? | [Modalities](/concepts/modalities) |
| Representation | How does the raw object become tokens, graph, coordinates, embedding, or features? | [Representation contract](/concepts/modalities/representation-contract), [Tensor shape notation](/concepts/math/tensor-shape-notation) |
| Graph construction | What nodes, edges, pair features, and inference-available context define the graph? | [Graph construction](/concepts/architectures/graph-construction), [Graph neural networks](/concepts/architectures/gnn), [Graph Transformer](/concepts/architectures/graph-transformer) |
| Task | What output space and validity rule define success? | [Tasks](/concepts/tasks) |
| Coordinate contract | What frame, symmetry, mapping, loss, and metric define coordinate claims? | [Coordinate modeling contract](/concepts/geometric-deep-learning/coordinate-modeling-contract) |
| Data | What example, label, split, and preprocessing contract define the dataset? | [Data](/concepts/data) |
| Architecture | What inductive bias and complexity does the model use? | [Architectures](/concepts/architectures), [AI architectures](/ai/architectures) |
| Learning method | What supervision, pretraining, transfer, or preference signal is used? | [Learning methods](/concepts/learning), [AI learning methods](/ai/learning-methods) |
| Objective-metric link | Does the optimized loss, reward, score, energy, or constraint support the reported metric and utility claim? | [Objective-metric alignment](/concepts/machine-learning/objective-metric-alignment) |
| Model selection | Which checkpoint, hyperparameter, threshold, seed, prompt, filter, or decoding setting was selected? | [Model selection](/concepts/machine-learning/model-selection) |
| Generative model | What distribution, score, energy, velocity, or sampling path is modeled? | [Generative models](/concepts/generative-models), [AI generative models](/ai/generative-models) |
| Math | Which formula, object type, distribution, constraint, or estimator is needed? | [Math](/math), [Math foundations](/concepts/math), [Constrained optimization](/concepts/math/constrained-optimization) |
| Evaluation | Which metric, baseline, split, uncertainty, and failure mode support the claim? | [Evaluation](/ai/evaluation), [Evaluation concepts](/concepts/evaluation) |
| Uncertainty and calibration | Is the result larger than uncertainty, and are probabilities meaningful? | [Confidence interval](/concepts/evaluation/confidence-interval), [Calibration](/concepts/evaluation/calibration), [Cross-validation](/concepts/evaluation/cross-validation) |
| Claim boundary | What exactly does the evidence prove, and where does it stop? | [Claim-evidence boundary](/concepts/evaluation/claim-evidence-boundary) |
| Systems | Does the claim depend on training, serving, tools, reproducibility, or compute? | [AI systems](/ai/systems), [Infra](/infra) |
| Scaling | Does quality depend on data size, model size, compute, memory, latency, or inference budget? | [Scaling claim contract](/concepts/systems/scaling-claim-contract) |
| Agent workflow | Does an LLM use tools, memory, planning, or verification? | [Agents](/agents) |

## Computational Biology Coverage

| Topic | Must Cover | Route |
| --- | --- | --- |
| Molecule or ligand model | standardization, identity, chemical state, representation, scaffold split, property or interaction label | [Molecules](/molecular-modeling/molecules), [Chemical state contract](/concepts/molecular-modeling/chemical-state-contract) |
| Protein model | sequence, structure, PLM/MSA/template policy, domains, residue indexing, family split, representation | [Proteins](/molecular-modeling/proteins), [Protein language model](/concepts/protein-modeling/protein-language-model) |
| Protein-ligand complex | protein context, ligand context, pocket, pose, interaction, split on both axes | [Structure-based modeling](/molecular-modeling/structure-based) |
| Docking or virtual screening | receptor/ligand preparation, pose generation, scoring, filtering, enrichment metric | [Docking](/molecular-modeling/docking) |
| Geometry-heavy model | coordinate frame, invariant/equivariant target, rotation/translation behavior, leakage risk | [Computational biology geometry](/molecular-modeling/geometry) |
| Genome sequence model | region, k-mer, variant, sequence context, split and annotation boundary | [Genome](/molecular-modeling/genome) |
| Computational biology benchmark | label semantics, assay/source, split unit, leakage, baseline, metric | [Data and evaluation](/molecular-modeling/data-evaluation) |
| Benchmark trap | negative construction, activity cliffs, applicability domain, assay harmonization, measurement ceiling | [Negative set](/concepts/evaluation/negative-set), [Activity cliff](/concepts/evaluation/activity-cliff), [Applicability domain](/concepts/evaluation/applicability-domain), [Assay harmonization](/concepts/evaluation/assay-harmonization), [Boltzmann ceiling analysis](/concepts/evaluation/boltzmann-ceiling) |

## Minimum Note Bundle

Before promoting a paper cluster into a Korean post, check that the bundle has:

- One route page: [[ai/paper-intake|AI paper intake]], [[molecular-modeling/paper-intake|Computational Biology paper intake]], or [[math/formula-intake|Formula intake]].
- One claim pattern when the paper is in computational biology: [[molecular-modeling/paper-claim-patterns|Computational Biology paper claim patterns]].
- One wiki-note quality pass before a concept becomes a reusable anchor: [[concepts/wiki-note-quality-gate|Wiki note quality gate]].
- One data or benchmark page when a score is central: [[concepts/data/benchmark-intake|Benchmark intake]].
- One topic-map contract when a cluster spans several axes: [[concepts/topic-map-contract|Topic map contract]].
- One object/modality/task page when the input or output is nontrivial.
- One representation contract when preprocessing or featurization changes the object seen by the model.
- One tensor-shape pass when a formula depends on batch, token, graph, coordinate, head, or candidate axes.
- One formula-pattern pass when the paper introduces or relies on an objective or metric: [[math/formula-patterns|Formula pattern catalog]].
- One constraint pass when feasibility, validity, projection, repair, or filtering is central: [[concepts/math/constrained-optimization|Constrained optimization]].
- One energy/score pass when a method uses energy, force, score matching, or Langevin-style sampling: [[concepts/generative-models/energy-based-model|Energy-based model]].
- One coordinate modeling contract when a paper predicts poses, conformers, structures, vectors, forces, or coordinate updates.
- One architecture or learning-method page when the method is central.
- One objective-metric alignment check when the training loss and reported metric differ.
- One evaluation page explaining metric, split, baseline, uncertainty, or failure mode.
- One scaling claim pass when model size, data size, compute, memory, latency, or throughput is part of the contribution.
- One claim boundary when the result could be overread: [[concepts/evaluation/claim-evidence-boundary|Claim-evidence boundary]].
- One paper note or paper-analysis note when a specific paper claim is being discussed.
- One claim-routing pass when a paper or post candidate spans multiple axes: [[papers/workflows/claim-routing|Claim routing]].
- One paper-to-wiki extraction pass before updating several support notes: [[papers/workflows/paper-to-wiki-extraction|Paper to wiki extraction]].
- One AI claim-pattern pass when the paper's method contribution is central: [[ai/paper-claim-patterns|AI paper claim patterns]].
- One cross-axis contract when AI, computational biology, and Math all matter: [[concepts/ai-computational-biology-math-contract|AI Computational Biology Math contract]].
- One concept-update pass when a paper changes reusable definitions, formulas, contracts, or evidence boundaries: [[papers/workflows/concept-update-contract|Concept update contract]].
- One readiness gate pass before promoting a multi-axis candidate: [[papers/workflows/ai-molecular-math-readiness-gate|AI Computational Biology Math readiness gate]].
- One fillable template pass for mixed AI/computational biology/Math papers: [[papers/workflows/ai-molecular-math-paper-template|AI Computational Biology Math paper template]].
- One Korean post intake pass when the result is reader-facing: [[posts/ai-molecular-math-post-intake|AI Computational Biology Math post intake]].
- One post bundle pass before writing a synthesis post: [[posts/wiki-bundle-checklist|Wiki bundle checklist]].

## Checks

- Is the central axis clear: object, method, formula, benchmark, paper cluster, or project?
- Are AI, computational biology, and Math links separated instead of collapsed into one vague topic?
- Does every performance claim have a route to benchmark intake or evaluation protocol?
- Does every computational biology claim name object, label, split, and leakage risk?
- Does every formula-heavy claim define symbols and the sampled distribution?
- Does every constrained or filtered-generation claim count invalid outputs?
- Does every probability claim distinguish accuracy, ranking, probability metric, and calibration?
- Is the public boundary clear before adding a post or paper note?

## Related

- [[concepts/index|Concepts]]
- [[concepts/wiki-note-quality-gate|Wiki note quality gate]]
- [[concepts/ai-computational-biology-math-contract|AI Computational Biology Math contract]]
- [[ai/index|AI]]
- [[molecular-modeling/index|Computational Biology]]
- [[math/index|Math]]
- [[papers/index|Papers]]
- [[concepts/topic-map-contract|Topic map contract]]
- [[papers/workflows/paper-to-wiki-extraction|Paper to wiki extraction]]
- [[posts/ai-molecular-math-post-intake|AI Computational Biology Math post intake]]
- [[posts/wiki-bundle-checklist|Wiki bundle checklist]]
- [[papers/workflows/ai-molecular-math-readiness-gate|AI Computational Biology Math readiness gate]]
- [[papers/workflows/concept-update-contract|Concept update contract]]
