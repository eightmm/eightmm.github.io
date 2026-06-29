---
title: Reproducibility Infra
tags:
  - infra
  - reproducibility
  - research-engineering
---

# Reproducibility Infra

Reproducibility infra note는 private system을 노출하지 않으면서 run을 이해하는 데 필요한 기록을 설명합니다.

Reproducibility는 question, run, artifact, claim 사이의 contract입니다.

$$
(\text{question}, \text{run}, \text{artifact}, \text{verification})
\rightarrow
\text{claim boundary}
$$

Public note는 private path, hostname, unpublished metric, internal task name을 노출하지 않으면서 boundary를 inspectable하게 만들어야 합니다.

## Scope

- run record와 artifact manifest.
- checkpoint, config, seed, environment, data-version boundary.
- public-safe module, container, package, runtime record.
- interrupted 또는 long-running job 이후의 reconciliation.
- completed, failed, superseded, inconclusive run을 구분하는 public note.

## Run Record Minimum

| 항목 | 공개 가능한 형태 |
| --- | --- |
| Question | what the run was trying to answer |
| Code state | commit hash or public diff summary |
| Data state | dataset version, split name, preprocessing version |
| Config | model, objective, important hyperparameters |
| Environment | package/container/module summary without private paths |
| Artifact | checkpoint/log/result type, not private absolute path |
| Verification | build, test, metric, smoke run, or failed check |
| Status | complete, failed, interrupted, superseded, inconclusive |

## 노트

- [[infra/reproducibility/run-record|Reproducible run record]]
- [[concepts/systems/environment-modules-containers|Environment modules and containers]]

## 확인할 것

- 두 run을 비교할 만큼 run identity가 명확한가?
- private path를 공개하지 않고 artifact type을 적었는가?
- split, seed, config, environment boundary가 기록되었는가?
- 해당 run이 public claim을 support하는지, private diagnosis에만 해당하는지 적었는가?
- failed 또는 interrupted run이 사라지지 않고 정직하게 표현되는가?

## 새 노트 위치

- 일반 run record는 여기에 둡니다.
- reproducibility가 핵심이면 environment capture와 module/container note도 여기에 둡니다.
- paper artifact availability는 [[papers/reproducibility/index|Paper reproducibility]]에 둡니다.
- experiment design과 evidence interpretation은 [[concepts/research-methodology/index|Research methodology]]에 둡니다.
- storage-specific artifact problem은 [[infra/io/index|Storage and IO]]에 둡니다.

## Related

- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/systems/environment-management|Environment management]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/research-methodology/experiment-ledger|Experiment ledger]]
