---
title: Wiki Note Quality Gate
tags:
  - concepts
  - wiki
  - workflow
---

# Wiki Note Quality Gate

Wiki note quality gate defines the minimum standard for reusable notes in this site. It applies to Concepts, Entities, AI, Computational Biology, Math, Papers workflows, Agents, Infra, and Projects when a page is meant to be reused by future posts or paper notes.

$$
\text{useful note}
=
\text{definition}
\land
\text{boundary}
\land
\text{contract}
\land
\text{checks}
\land
\text{links}
$$

## Minimum Fields

| Field | Pass When |
| --- | --- |
| Definition | the note states what the concept is in one short paragraph |
| Boundary | the note states what is included and what is out of scope |
| Formula or rule | a formula, transform, workflow, or decision rule is included when it clarifies the concept |
| Contract | object, representation, task, objective, metric, split, artifact, or tool fields are explicit when relevant |
| Checks | the note has practical questions that catch common misuse |
| Related links | the note links to upstream and downstream concepts |
| Uncertainty | missing facts are marked `to verify` instead of invented |
| Public boundary | private paths, accounts, infrastructure details, unpublished results, and collaborator details are absent |

## Note Types

| Note Type | Must Emphasize |
| --- | --- |
| AI method | input/output, architecture or learning signal, objective, evaluation risk |
| Molecular modeling | entity, representation, preprocessing, split, leakage, metric |
| Math | symbols, shapes, distribution, operation, estimator, claim relation |
| Data/evaluation | example unit, split unit, metric, baseline, uncertainty, failure mode |
| Paper workflow | source status, claim boundary, evidence, artifact, next decision |
| Agent workflow | state, tool contract, verifier, handoff, completion evidence |
| Infra note | symptom, public-safe evidence, safe action, prevention, boundary |
| Project note | problem, artifact, design decision, verification, status, next work |

## Skeleton

```markdown
# Concept Name

One-paragraph definition.

## Core Rule

Formula, workflow, or decision rule if useful.

## Contract

| Field | Meaning |
| --- | --- |
| to verify | to verify |

## Checks

- Question that catches misuse.
- Question that clarifies evaluation or boundary.

## Related

- [[concepts/index|Concepts]]
```

## Stop Conditions

- The page is only a title and link list.
- The page repeats another note without adding a clearer boundary.
- The page uses a model name without task, data, objective, or evaluation context.
- The page contains a formula without symbol definitions.
- The page contains a benchmark claim without split, metric, baseline, or uncertainty.
- The page contains operational details that could expose private systems.

## Related

- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
- [[concepts/coverage-matrix|Coverage matrix]]
- [[concepts/topic-map-contract|Topic map contract]]
- [[math/formula-explanation-ladder|Formula explanation ladder]]
- [[papers/workflows/concept-update-contract|Concept update contract]]
- [[posts/post-promotion-gate|Post promotion gate]]
