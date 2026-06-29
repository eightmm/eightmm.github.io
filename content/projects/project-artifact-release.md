---
title: Project Artifact Release
tags:
  - projects
  - artifacts
  - reproducibility
---

# Project Artifact Release

Project artifact release는 project에서 무엇을 공개할 수 있고 무엇을 private로 남겨야 하는지 설명합니다. Code, model weight, config, log, dataset, generated output, diagram, runbook, blog post에 적용합니다.

Release decision은 아래처럼 쓸 수 있습니다.

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

여기서 $a$는 artifact입니다.

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

- Public-safe: credential, private path, hostname, account name, SSH detail, collaborator context, unpublished sensitive result가 없습니다.
- License-safe: code, data, figure, model weight를 redistribute 또는 link할 수 있습니다.
- Reproducible: version, config, environment, verification이 적혀 있습니다.
- Useful: reader가 project를 run, inspect, compare, understand하는 데 도움이 됩니다.
- Bounded: intended use와 out-of-scope use가 explicit합니다.

## Missing or Private Artifacts

Artifact를 공개할 수 없다면 public-safe language로 이유를 적습니다.

| Status | Meaning |
| --- | --- |
| `not released` | artifact는 있지만 public이 아님 |
| `to verify` | public status가 아직 확인되지 않음 |
| `not applicable` | 이 project에 해당 artifact가 필요하지 않음 |
| `replaced by summary` | sanitized summary만 public임 |

## Related

- [[projects/project-lifecycle|Project lifecycle]]
- [[projects/project-note-format|Project note format]]
- [[concepts/systems/model-card|Model card]]
- [[concepts/systems/inference-contract|Inference contract]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/data/dataset-card|Dataset card]]
- [[logs/sanitization-checklist|Sanitization checklist]]
