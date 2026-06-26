---
title: Slurm Job Script
tags:
  - infra
  - hpc
  - slurm
---

# Slurm Job Script

A Slurm job script should make resource assumptions, environment setup, logs, and failure behavior explicit. Public examples should use placeholders and avoid private partitions, accounts, hostnames, paths, or queue names.

The script is the executable contract for a run:

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

## Generic Template

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

## Safer Template Shape

For longer jobs, make the run directory, config, and checkpoint behavior explicit:

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

Use placeholders in public notes. Do not publish private partition names, account names, node names, hostnames, internal paths, SSH details, or project-specific allocations.

## Failure Behavior

`set -euo pipefail` catches common script failures:

- `-e`: stop on command failure.
- `-u`: stop on unset variables.
- `-o pipefail`: fail a pipeline if any command fails.

This does not replace application-level error handling. Training code still needs explicit checkpoint, resume, and artifact validation logic.

## Log Contract

Logs should identify:

$$
(\text{job id},\ \text{job name},\ \text{config},\ \text{commit},\ \text{start time},\ \text{exit state})
$$

The public version can describe these fields without exposing private paths or cluster-specific identifiers.

## Checks

- Does the script fail fast with `set -euo pipefail`?
- Are logs named with job name and job id?
- Are CPU, memory, GPU, and wall-time assumptions explicit?
- Does the command run from the submitted project directory?
- Is resume behavior documented for jobs that may hit time limits?
- Are output, checkpoint, and artifact directories created before the main command?
- Are environment activation and dependency versions reproducible?
- Is the script generic enough for public notes, with private cluster values removed?
- Is completion checked with an artifact or manifest, not only a successful process exit?

## Related

- [[infra/hpc/slurm|Slurm]]
- [[infra/hpc/resource-request|Resource request]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[projects/hpc-research-workflows|HPC research workflows]]
