---
title: Topic Map Contract
tags:
  - concepts
  - workflow
  - posts
---

# Topic Map Contract

Topic map contract defines the minimum structure for a map-style wiki note or Korean post. Use it when a topic spans AI, Computational Biology, Math, Papers, Infra, or Agents and needs a reader-facing route rather than one isolated definition.

$$
\text{topic map}
=
\text{axis}
+ \text{objects}
+ \text{methods}
+ \text{objectives}
+ \text{evidence}
+ \text{next path}
$$

## Map Fields

| Field | Question |
| --- | --- |
| Reader question | What question does the map help answer? |
| Primary axis | Is the map centered on object, modality, task, architecture, learning method, formula, benchmark, system, or agent workflow? |
| Object boundary | What entities or inputs are included, and what is out of scope? |
| Representation | How are raw objects converted into model inputs? |
| Task/output | What output space and validity rule define success? |
| Method family | Which architectures, learning signals, generative processes, or agent loops matter? |
| Math anchor | Which formulas, distributions, symmetries, or estimators are necessary? |
| Evidence boundary | What metric, split, baseline, artifact, or benchmark supports claims? |
| Link bundle | Which reference notes should readers follow next? |
| Public boundary | What private context, unpublished result, or internal detail is intentionally absent? |

## Axis Examples

| Map Type | Primary Axis | Required Secondary Axes |
| --- | --- | --- |
| Architecture map | architecture | modality, complexity, learning signal, evaluation |
| Molecular modeling map | object/workflow | representation, split, metric, leakage |
| Math map | formula/objective | symbols, distribution, estimator, metric |
| Paper cluster map | claim/evidence | paper notes, benchmark contract, concept updates |
| Project map | artifact/problem | design, verification, public boundary |
| Infra map | system behavior | resource, failure mode, reproducibility, safe runbook |
| Agent map | workflow | state, tool contract, verifier, handoff, evidence ledger |

## Minimum Link Bundle

A useful topic map should link at least one note from most relevant groups:

- Object/entity or modality note.
- Task or output-space note.
- Method or architecture note.
- Math or formula note.
- Data, split, or benchmark note.
- Evaluation or claim-boundary note.
- Paper, project, infra, or agent workflow note when relevant.

## Stop Conditions

- The map is only a list of links without a reader question.
- The map hides the primary axis.
- The map mixes AI, Computational Biology, and Math without saying which one owns the explanation.
- The map contains paper-specific claims without evidence boundary.
- The map repeats full definitions that should live in concept notes.
- The map depends on private data, unpublished results, or internal infrastructure context.

## Related

- [[concepts/coverage-matrix|Coverage matrix]]
- [[posts/post-promotion-gate|Post promotion gate]]
- [[posts/wiki-to-post-workflow|Wiki to post workflow]]
- [[papers/workflows/claim-routing|Claim routing]]
- [[papers/workflows/concept-update-contract|Concept update contract]]
- [[papers/workflows/ai-molecular-math-readiness-gate|AI-Molecular-Math readiness gate]]
