---
title: Slurm Job Script
tags:
  - infra
  - hpc
  - slurm
---

# Slurm Job Script

Slurm job script는 resource assumption, environment setup, log, failure behavior를 explicit하게 만들어야 합니다. Public example은 placeholder를 사용하고 private partition, account, hostname, path, queue name을 피해야 합니다.

Script는 run을 위한 executable contract입니다.

$$
\text{job script}
=
\text{resources}
+
\text{environment}
+
\text{command}
+
\text{logging}
+
\text{resume policy}
$$

## Generic template

```bash
#!/usr/bin/env bash
#SBATCH --job-name=example-job
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

python train.py --config configs/example.yaml
```

## 더 안전한 template 형태

긴 job에서는 run directory, config, checkpoint behavior를 explicit하게 둡니다.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=example-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=<hh:mm:ss>
#SBATCH --cpus-per-task=<cpu-count>
#SBATCH --mem=<memory>
#SBATCH --gres=gpu:<gpu-count>

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints artifacts

echo "job_id=${SLURM_JOB_ID}"
echo "submit_dir=${SLURM_SUBMIT_DIR}"
echo "started_at=$(date -Is)"

python train.py \
  --config configs/example.yaml \
  --output-dir artifacts/example \
  --checkpoint-dir checkpoints/example

echo "finished_at=$(date -Is)"
```

Public note에서는 placeholder를 씁니다. Private partition name, account name, node name, hostname, internal path, SSH detail, project-specific allocation을 공개하지 않습니다.

## Failure behavior

`set -euo pipefail`은 흔한 script failure를 잡습니다.

- `-e`: command failure에서 멈춥니다.
- `-u`: unset variable에서 멈춥니다.
- `-o pipefail`: pipeline 안의 command 하나라도 실패하면 pipeline을 실패로 처리합니다.

이 설정은 application-level error handling을 대체하지 않습니다. Training code에는 여전히 explicit checkpoint, resume, artifact validation logic이 필요합니다.

## Log contract

Log는 아래를 식별해야 합니다.

$$
(\text{job id},\ \text{job name},\ \text{config},\ \text{commit},\ \text{start time},\ \text{exit state})
$$

Public version은 private path나 cluster-specific identifier를 노출하지 않고 이 field를 설명할 수 있습니다.

## Common `sbatch` Options

Use this as a public-safe reference. Replace real partition, node, account, path, and email values with placeholders.

| Option | Meaning | Public note |
| --- | --- | --- |
| `--job-name=<name>` / `-J <name>` | human-readable job name | avoid internal project names |
| `--partition=<partition>` | scheduler partition or queue | use `<partition>` unless policy is public |
| `--time=<hh:mm:ss>` | wall-time limit | tie to measured smoke-run estimate |
| `--nodes=<count>` / `-N <count>` | node count | justify multi-node request |
| `--ntasks-per-node=<count>` | processes per node | match launcher and rank layout |
| `--cpus-per-task=<cores>` | CPU cores per task | match dataloader/preprocessing needs |
| `--mem=<limit>` | memory per node | use generic memory class in public notes |
| `--mem-per-cpu=<memory>` | memory per CPU | do not combine blindly with `--mem` |
| `--gres=gpu:<count>` | generic GPU request | prefer generic count over private GPU class names |
| `--nodelist=<node>` / `-w <node>` | specific node selection | avoid publishing node names |
| `--nodefile=<file>` / `-F <file>` | node list file | avoid private file paths |
| `--output=<path>` | stdout path | path directory must exist |
| `--error=<path>` | stderr path | path directory must exist |
| `--export=<vars>` | environment variables passed to job | never export secrets |
| `--dependency=<rule>:<job-id>` | job dependency | record why the dependency exists |
| `--mail-type=<events>` | email notification events | avoid publishing email addresses |
| `--mail-user=<email>` | notification recipient | omit from public examples |
| `--begin=<date/time>` | delayed start | use placeholder date/time |
| `--exclusive` | exclusive node allocation | justify because it reduces sharing |

Dependency patterns:

```bash
# Run after another job starts
#SBATCH --dependency=after:<job-id>

# Run only if another job succeeds
#SBATCH --dependency=afterok:<job-id>

# Run if another job fails
#SBATCH --dependency=afternotok:<job-id>

# Run after another job finishes in any state
#SBATCH --dependency=afterany:<job-id>
```

## Option Grouping

| Group | Options | Main risk |
| --- | --- | --- |
| identity | `--job-name`, `--output`, `--error` | leaking project names or private paths |
| resource | `--nodes`, `--ntasks-per-node`, `--cpus-per-task`, `--mem`, `--gres` | oversized requests increase queue wait |
| placement | `--partition`, `--nodelist`, `--nodefile`, `--exclusive` | exposing cluster topology or forcing fragile placement |
| environment | `--export` | leaking secrets or environment-specific assumptions |
| workflow | `--dependency`, `--begin`, `--mail-type` | hidden ordering assumptions |

## 확인할 것

- script가 `set -euo pipefail`로 빠르게 실패하는가?
- log가 job name과 job id로 이름 붙는가?
- CPU, memory, GPU, wall-time assumption이 explicit한가?
- command가 submitted project directory에서 실행되는가?
- time limit에 걸릴 수 있는 job의 resume behavior가 문서화되어 있는가?
- main command 전에 output, checkpoint, artifact directory를 생성하는가?
- environment activation과 dependency version이 reproducible한가?
- private cluster value를 제거한 public note용 generic script인가?
- completion을 successful process exit만이 아니라 artifact 또는 manifest로 확인하는가?

## Related

- [[infra/hpc/slurm|Slurm]]
- [[infra/hpc/resource-request|Resource request]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[projects/hpc-research-workflows|HPC research workflows]]
