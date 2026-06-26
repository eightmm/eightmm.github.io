---
title: Coverage Matrix
tags:
  - concepts
  - ai
  - bio-ai
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
| AI paper claim | architecture, objective, evidence, and system boundary | [AI paper intake](/ai/paper-intake) |
| Bio-AI paper claim | object, representation, label, split, leakage | [Bio-AI paper intake](/bio/paper-intake) |
| Formula or objective | symbols, distributions, derivatives, metrics | [Formula intake](/math/formula-intake) |
| Benchmark score | data, task, split, metric, allowed information, reporting | [Benchmark intake](/concepts/data/benchmark-intake) |
| Korean synthesis post | reader question, main axis, minimum formulas, wiki links | [AI-Bio-Math post intake](/posts/ai-bio-math-post-intake) |
| Paper review workflow | metadata, claims, evidence, artifacts, reproduction | [Paper review workflow](/papers/workflows/paper-review-workflow) |

## Axis Coverage

| Axis | Question | Canonical Area |
| --- | --- | --- |
| Object | What entity is being modeled? | [Entities](/entities), [Bio-AI entities](/bio/entities) |
| Modality | What form is the input or output? | [Modalities](/concepts/modalities) |
| Task | What output space and validity rule define success? | [Tasks](/concepts/tasks) |
| Data | What example, label, split, and preprocessing contract define the dataset? | [Data](/concepts/data) |
| Architecture | What inductive bias and complexity does the model use? | [Architectures](/concepts/architectures), [AI architectures](/ai/architectures) |
| Learning method | What supervision, pretraining, transfer, or preference signal is used? | [Learning methods](/concepts/learning), [AI learning methods](/ai/learning-methods) |
| Generative model | What distribution, score, velocity, or sampling path is modeled? | [Generative models](/concepts/generative-models), [AI generative models](/ai/generative-models) |
| Math | Which formula, object type, distribution, or estimator is needed? | [Math](/math), [Math foundations](/concepts/math) |
| Evaluation | Which metric, baseline, split, uncertainty, and failure mode support the claim? | [Evaluation](/ai/evaluation), [Evaluation concepts](/concepts/evaluation) |
| Systems | Does the claim depend on training, serving, tools, reproducibility, or compute? | [AI systems](/concepts/systems), [Infra](/infra) |
| Agent workflow | Does an LLM use tools, memory, planning, or verification? | [Agents](/agents) |

## Bio-AI Coverage

| Topic | Must Cover | Route |
| --- | --- | --- |
| Molecule or ligand model | standardization, identity, representation, scaffold split, property or interaction label | [Molecules](/bio/molecules) |
| Protein model | sequence, structure, domains, residue indexing, family split, representation | [Proteins](/bio/proteins) |
| Protein-ligand complex | protein context, ligand context, pocket, pose, interaction, split on both axes | [Structure-based AI](/bio/structure-based-ai) |
| Docking or virtual screening | receptor/ligand preparation, pose generation, scoring, filtering, enrichment metric | [Docking](/bio/docking) |
| Geometry-heavy model | coordinate frame, invariant/equivariant target, rotation/translation behavior, leakage risk | [Bio geometry](/bio/geometry) |
| Genome sequence model | region, k-mer, variant, sequence context, split and annotation boundary | [Genome](/bio/genome) |
| Bio benchmark | label semantics, assay/source, split unit, leakage, baseline, metric | [Data and evaluation](/bio/data-evaluation) |

## Minimum Note Bundle

Before promoting a paper cluster into a Korean post, check that the bundle has:

- One route page: [[ai/paper-intake|AI paper intake]], [[bio/paper-intake|Bio-AI paper intake]], or [[math/formula-intake|Formula intake]].
- One data or benchmark page when a score is central: [[concepts/data/benchmark-intake|Benchmark intake]].
- One object/modality/task page when the input or output is nontrivial.
- One architecture or learning-method page when the method is central.
- One evaluation page explaining metric, split, baseline, uncertainty, or failure mode.
- One paper note or paper-analysis note when a specific paper claim is being discussed.
- One Korean post intake pass when the result is reader-facing: [[posts/ai-bio-math-post-intake|AI-Bio-Math post intake]].

## Checks

- Is the central axis clear: object, method, formula, benchmark, paper cluster, or project?
- Are AI, Bio-AI, and Math links separated instead of collapsed into one vague topic?
- Does every performance claim have a route to benchmark intake or evaluation protocol?
- Does every Bio-AI claim name object, label, split, and leakage risk?
- Does every formula-heavy claim define symbols and the sampled distribution?
- Is the public boundary clear before adding a post or paper note?

## Related

- [[concepts/index|Concepts]]
- [[ai/index|AI]]
- [[bio/index|Bio-AI]]
- [[math/index|Math]]
- [[papers/index|Papers]]
- [[posts/ai-bio-math-post-intake|AI-Bio-Math post intake]]
