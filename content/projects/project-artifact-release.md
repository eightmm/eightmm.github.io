---
title: Project Artifact Release
tags:
  - projects
  - artifacts
  - reproducibility
---

# Project Artifact Release

A project artifact release describes what can be shared publicly from a project and what must remain private. It applies to code, model weights, configs, logs, datasets, generated outputs, diagrams, runbooks, and blog posts.

A release decision can be written as:

$$
\operatorname{release}(a)
=
\operatorname{safe}(a)
\land
\operatorname{licensed}(a)
\land
\operatorname{useful}(a)
\land
\operatorname{verifiable}(a)
$$

where $a$ is the artifact.

## Artifact Classes

| Artifact | Public release field |
| --- | --- |
| Code | repository, license, entry point, dependency boundary |
| Config | sanitized options, defaults, environment class |
| Model | model card, checkpoint status, intended use, limits |
| Inference workflow | input schema, output schema, errors, runtime envelope |
| Dataset | public source, license, preprocessing, split contract |
| Run output | metric definition, split, aggregation, public-safe summary |
| Runbook | generic steps, checks, public boundary |
| Diagram | self-drawn or licensed image, no private topology |

## Release Checklist

- Public-safe: no credentials, private paths, hostnames, account names, SSH details, collaborator context, or unpublished sensitive results.
- License-safe: code, data, figures, and model weights can be redistributed or linked.
- Reproducible: version, config, environment, and verification are stated.
- Useful: the artifact helps a reader run, inspect, compare, or understand the project.
- Bounded: intended use and out-of-scope use are explicit.

## Missing or Private Artifacts

If an artifact cannot be released, state why in public-safe language:

| Status | Meaning |
| --- | --- |
| `not released` | The artifact exists but is not public |
| `to verify` | Public status is unknown |
| `not applicable` | The artifact is not needed for this project |
| `replaced by summary` | Only a sanitized summary is public |

## Related

- [[projects/project-lifecycle|Project lifecycle]]
- [[projects/project-note-format|Project note format]]
- [[concepts/systems/model-card|Model card]]
- [[concepts/systems/inference-contract|Inference contract]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/data/dataset-card|Dataset card]]
- [[logs/sanitization-checklist|Sanitization checklist]]
