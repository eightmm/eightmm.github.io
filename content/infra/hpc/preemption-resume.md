---
title: Preemption and Resume
tags:
  - infra
  - hpc
  - reliability
---

# Preemption and Resume

Preemption은 scheduler가 higher-priority work에 resource를 필요로 하거나 time/resource limit에 도달해 running job이 completion 전에 멈추는 상황입니다. Resume은 saved state에서 이어서 실행하는 workflow입니다.

Long job에서는 아래처럼 봅니다.

$$
\text{progress}_{t+1}
= \operatorname{resume}(\text{checkpoint}_t, \text{inputs}, \text{config})
$$

Checkpoint에는 restart 이후 run이 조용히 다른 실험으로 바뀌지 않을 만큼 충분한 state가 들어 있어야 합니다.

## Resume Invariant

Resume은 새 실험이 아니라 같은 run의 continuation이어야 합니다.

$$
\operatorname{id}_{\mathrm{resume}}
=
H(\text{code}, \text{data}, \text{config}, \text{seed}, \text{checkpoint schema})
$$

Resume 전후 이 identity가 바뀌면 run을 continuation으로 기록하지 말고 superseded run으로 분리합니다.

| Must match | Why |
| --- | --- |
| code or compatible checkpoint schema | state tensor와 optimizer 구조가 맞아야 함 |
| data ordering or shard cursor | duplicate/skip 방지 |
| config and objective | 같은 experiment claim 유지 |
| seed and sampler state when needed | stochastic continuation 재현성 |
| output directory policy | partial/final artifact 충돌 방지 |

## 저장할 것

- training용 model weight와 optimizer state.
- step, epoch, random seed, scheduler state.
- data shard 또는 task index.
- config, code commit, environment summary.
- completion marker가 있는 partial output.

## Resume pattern

1. clean run record에서 시작합니다.
2. checkpoint를 atomic하게 쓰거나 temporary file과 rename을 사용합니다.
3. load 전에 checkpoint compatibility를 validate합니다.
4. latest complete checkpoint에서 resume합니다.
5. final output이 resumed run에서 나온 것인지 기록합니다.
6. run complete로 표시하기 전에 resumed job을 reconcile합니다.

## Partial Output Policy

Partial output은 final output과 분리합니다.

| Artifact | Safe policy |
| --- | --- |
| checkpoint | write temp file, fsync/rename if possible, keep latest complete pointer |
| metrics | append with step index and flush policy |
| generated samples | shard-level complete marker |
| merged result | build only from complete shard list |
| logs | keep interruption and resume event visible |

Completion marker가 없는 artifact는 final claim에 쓰지 않습니다.

## 확인할 것

- process를 잃은 뒤 job을 restart할 수 있는가?
- training일 때 checkpoint가 optimizer와 scheduler state를 포함하는가?
- partial output과 complete output을 구분할 수 있는가?
- resume path를 짧은 smoke run으로 test했는가?
- public documentation이 private job ID, path, queue, result detail을 피하는가?
- run record가 original launch, interruption, resume, final closeout을 구분하는가?
- resume smoke test가 실제 checkpoint load와 one-step write를 검증하는가?
- preemption 이후 repeated side effect나 duplicate output이 생기지 않는가?

## Related

- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/reproducibility/run-record|Reproducible run record]]
