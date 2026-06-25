---
title: Slurm Job Script
tags:
  - infra
  - hpc
  - slurm
---

# Slurm Job Script

A Slurm job script should make resource assumptions, environment setup, logs, and failure behavior explicit. Public examples should use placeholders and avoid private partitions, accounts, hostnames, paths, or queue names.

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

## Checks

- Does the script fail fast with `set -euo pipefail`?
- Are logs named with job name and job id?
- Are CPU, memory, GPU, and wall-time assumptions explicit?
- Does the command run from the submitted project directory?
- Is resume behavior documented for jobs that may hit time limits?

## Related

- [[infra/hpc/slurm|Slurm]]
- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[projects/hpc-research-workflows|HPC research workflows]]
