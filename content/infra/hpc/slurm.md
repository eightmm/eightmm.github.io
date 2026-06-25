---
title: Slurm
tags:
  - infra
  - hpc
  - slurm
---

# Slurm

Slurm is a workload manager used to submit, schedule, monitor, and cancel jobs on shared compute clusters. This page keeps public, non-sensitive workflow notes only.

## Public Checklist

- Use generic examples instead of private cluster names.
- Avoid account names, hostnames, SSH connection details, internal paths, and private partitions.
- Record resource assumptions without exposing project-specific allocations.
- Keep unpublished metrics and experiment results out of public notes.

## Generic Commands

```bash
sinfo
squeue
sbatch job.sbatch
scancel <job-id>
```

## Reproducibility Notes

- Capture code commit, environment, seed, and dataset version.
- Prefer small smoke tests before large jobs.
- Link public experiment methodology into [[agents/llm-wiki|LLM Wiki]] pages when it becomes reusable.

## Related

- [[research/protein-modeling/mambafold|MambaFold]]
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
- [[projects/index|Project index]]
- [[infra/index|Infra]]
