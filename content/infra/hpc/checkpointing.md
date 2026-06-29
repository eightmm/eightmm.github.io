---
title: Checkpointing
tags:
  - infra
  - hpc
  - training
---

# Checkpointing

Checkpointing은 preemption, failure, wall-clock limit 이후 job을 resume할 수 있도록 training state를 주기적으로 저장하는 방식입니다. Checkpoint는 weight만이 아니라 run 전체를 정확히 restore해야 합니다.

Checkpoint state는 아래처럼 쓸 수 있습니다.

$$
C_t
=
(\theta_t,\ o_t,\ s_t,\ k_t,\ r_t,\ e_t)
$$

여기서 $\theta_t$는 model state, $o_t$는 optimizer state, $s_t$는 scheduler state, $k_t$는 step 또는 epoch, $r_t$는 RNG state, $e_t$는 environment/run metadata입니다.

## Resume contract

Resume은 아래 조건을 만족해야 합니다.

$$
\operatorname{resume}(C_t)
\rightarrow
\text{same training state at step } t
$$

즉 next batch, learning-rate schedule, gradient-scaler state, distributed rank behavior, random augmentation이 의도한 run policy와 일관되어야 합니다.

## Atomic Write

Checkpoint write는 latest checkpoint를 corrupt하지 않아야 합니다.

```text
write checkpoint.tmp
fsync checkpoint.tmp
rename checkpoint.tmp -> checkpoint.latest
write manifest.json
```

구체 구현은 달라질 수 있지만 원칙은 안정적입니다. Crash 이후에는 old valid checkpoint 또는 new valid checkpoint 중 하나가 남아야 하며, half-written file이 남으면 안 됩니다.

## 실전 check

- model weight, optimizer state, scheduler state, step count, RNG seed를 저장합니다.
- crash가 latest checkpoint를 corrupt하지 않도록 atomic write를 사용합니다.
- disk 사용량을 제한하기 위해 rolling window와 periodic milestone을 함께 둡니다.
- 긴 run에 믿고 쓰기 전에 작은 run에서 resume을 test합니다.
- checkpoint와 함께 code commit, environment, dataset version을 기록합니다.
- reconciliation 중에는 compatibility check를 통과한 checkpoint만 resumable로 취급합니다.
- fp16 training을 쓰면 mixed-precision scaler state를 저장합니다.
- distributed training에서는 world size, sharding, sampler position, rank-local state를 resume할 만큼 충분히 저장합니다.
- loaded checkpoint가 current config와 code expectation에 맞는지 validate합니다.
- final artifact와 transient recovery checkpoint를 분리합니다.

## Cadence

Checkpoint interval은 overhead와 lost work 사이의 tradeoff입니다.

$$
\operatorname{expected\ lost\ work}
\approx
\frac{\Delta t_{\mathrm{ckpt}}}{2}
$$

여기서 $\Delta t_{\mathrm{ckpt}}$는 checkpoint 사이의 시간입니다. Checkpoint가 너무 잦으면 IO가 지배하고, 너무 드물면 preemption이나 timeout 때 compute 낭비가 커집니다.

## Completion marker

Long job에서 checkpoint는 completed output과 같지 않습니다. Explicit marker 또는 manifest를 사용합니다.

$$
\text{complete}
\ne
\text{latest checkpoint exists}
$$

Manifest는 private path를 노출하지 않고 final step, expected shard count, config hash, artifact type을 식별해야 합니다.

## Compatibility check

Resume 전에 아래를 비교합니다.

- Config hash.
- Model architecture version.
- Dataset and split version.
- Optimizer and scheduler type.
- Precision mode.
- Distributed/sharding policy.
- Code commit or release identifier.

이 값들이 맞지 않으면 run이 crash 없이 load되더라도 다른 experiment로 이어질 수 있습니다.

## Related

- [[infra/hpc/slurm|Slurm]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]
- [[infra/hpc/preemption-resume|Preemption and resume]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[concepts/systems/distributed-training-runbook|Distributed training]]
- [[infra/index|Infra]]
