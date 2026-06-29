---
title: Paper Reproducibility
unlisted: true
tags:
  - papers
  - reproducibility
---

# Paper Reproducibility

Paper reproducibility note는 특정 paper claim을 rerun, reimplement, compare할 만큼 public material이 충분한지 판단합니다.

Reproduction은 paper 전체가 아니라 claim 하나에 scope를 맞춰야 합니다.

$$
\operatorname{ready}(p,c)
=
\operatorname{artifacts}(p)
\land
\operatorname{spec}(c)
\land
\operatorname{feasible}(c)
\land
\operatorname{verifiable}(c)
$$

여기서 $p$는 paper이고 $c$는 확인할 claim입니다.

## Scope

- public artifact availability.
- reproducibility checklist와 implementation readiness.
- minimum reproduction plan과 reproduction-result record.
- rerun, reimplementation, diagnostic check를 위한 public-safe evidence.

## 노트

- [[papers/reproducibility/artifact-availability|Artifact availability]]
- [[papers/reproducibility/checklist|Reproducibility checklist]]
- [[papers/reproducibility/implementation-readiness|Implementation readiness]]
- [[papers/reproducibility/reproduction-plan|Reproduction plan]]
- [[papers/reproducibility/reproduction-result|Reproduction result]]

## 확인할 것

- code, data, split, config, weight, log, prediction, environment를 분리해서 확인했는가?
- target claim이 public artifact로 test할 수 있을 만큼 좁은가?
- compute를 쓰기 전에 minimum viable experiment가 정의되었는가?
- result가 success, contradiction, inconclusive outcome, diagnostic-only value 중 무엇인지 적는가?
- private dataset, private path, unpublished metric, collaborator detail을 제외했는가?

## 새 노트 위치

- paper-specific artifact와 reproduction note는 여기에 둡니다.
- 일반 experiment design은 [[concepts/research-methodology/minimum-viable-experiment|Minimum viable experiment]]에 둡니다.
- run artifact structure는 [[concepts/systems/run-artifact|Run artifact]]에 둡니다.
- public operational run record는 [[infra/reproducibility/index|Reproducibility infra]]에 둡니다.

## Related

- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/research-methodology/minimum-viable-experiment|Minimum viable experiment]]
- [[papers/analysis/index|Paper analysis]]
