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
| AI paper claim | architecture, objective, evidence, and system boundary | [AI paper intake](/ai/paper-intake) |
| Molecular modeling paper claim | object, representation, label, split, leakage | [Molecular modeling paper intake](/bio/paper-intake) |
| Formula or objective | symbols, distributions, derivatives, metrics | [Formula intake](/math/formula-intake) |
| Benchmark score | data, task, split, metric, allowed information, reporting | [Benchmark intake](/concepts/data/benchmark-intake) |
| Korean synthesis post | reader question, main axis, minimum formulas, wiki links | [AI-Molecular-Math post intake](/posts/ai-bio-math-post-intake) |
| Paper review workflow | metadata, claims, evidence, artifacts, reproduction | [Paper review workflow](/papers/workflows/paper-review-workflow) |
| Claim route | primary axis, secondary axes, bucket, concept updates | [Claim routing](/papers/workflows/claim-routing) |
| Fillable paper note | reusable skeleton for AI/molecular modeling/Math papers | [AI-Molecular-Math paper template](/papers/workflows/ai-bio-math-paper-template) |
| Paper bucket | decide whether the paper belongs in SBDD, protein modeling, architecture, generation, learning methods, or systems | [Paper triage](/papers/workflows/paper-triage) |

## Axis Coverage

| Axis | Question | Canonical Area |
| --- | --- | --- |
| Object | What entity is being modeled? | [Entities](/entities), [Molecular modeling entities](/bio/entities) |
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

## Molecular Modeling Coverage

| Topic | Must Cover | Route |
| --- | --- | --- |
| Molecule or ligand model | standardization, identity, representation, scaffold split, property or interaction label | [Molecules](/bio/molecules) |
| Protein model | sequence, structure, domains, residue indexing, family split, representation | [Proteins](/bio/proteins) |
| Protein-ligand complex | protein context, ligand context, pocket, pose, interaction, split on both axes | [Structure-based modeling](/bio/structure-based-ai) |
| Docking or virtual screening | receptor/ligand preparation, pose generation, scoring, filtering, enrichment metric | [Docking](/bio/docking) |
| Geometry-heavy model | coordinate frame, invariant/equivariant target, rotation/translation behavior, leakage risk | [Bio geometry](/bio/geometry) |
| Genome sequence model | region, k-mer, variant, sequence context, split and annotation boundary | [Genome](/bio/genome) |
| Bio benchmark | label semantics, assay/source, split unit, leakage, baseline, metric | [Data and evaluation](/bio/data-evaluation) |

## Minimum Note Bundle

Before promoting a paper cluster into a Korean post, check that the bundle has:

- One route page: [[ai/paper-intake|AI paper intake]], [[bio/paper-intake|Molecular modeling paper intake]], or [[math/formula-intake|Formula intake]].
- One data or benchmark page when a score is central: [[concepts/data/benchmark-intake|Benchmark intake]].
- One object/modality/task page when the input or output is nontrivial.
- One architecture or learning-method page when the method is central.
- One evaluation page explaining metric, split, baseline, uncertainty, or failure mode.
- One paper note or paper-analysis note when a specific paper claim is being discussed.
- One claim-routing pass when a paper or post candidate spans multiple axes: [[papers/workflows/claim-routing|Claim routing]].
- One fillable template pass for mixed AI/molecular modeling/Math papers: [[papers/workflows/ai-bio-math-paper-template|AI-Molecular-Math paper template]].
- One Korean post intake pass when the result is reader-facing: [[posts/ai-bio-math-post-intake|AI-Molecular-Math post intake]].

## Checks

- Is the central axis clear: object, method, formula, benchmark, paper cluster, or project?
- Are AI, molecular modeling, and Math links separated instead of collapsed into one vague topic?
- Does every performance claim have a route to benchmark intake or evaluation protocol?
- Does every molecular modeling claim name object, label, split, and leakage risk?
- Does every formula-heavy claim define symbols and the sampled distribution?
- Is the public boundary clear before adding a post or paper note?

## Related

- [[concepts/index|Concepts]]
- [[ai/index|AI]]
- [[bio/index|Molecular Modeling]]
- [[math/index|Math]]
- [[papers/index|Papers]]
- [[posts/ai-bio-math-post-intake|AI-Molecular-Math post intake]]
