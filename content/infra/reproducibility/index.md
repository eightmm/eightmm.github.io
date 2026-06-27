---
title: Reproducibility Infra
tags:
  - infra
  - reproducibility
  - research-engineering
---

# Reproducibility Infra

Reproducibility infra notes describe the records needed to understand runs without exposing private systems.

Reproducibility is a contract between the question, run, artifact, and claim:

$$
(\text{question}, \text{run}, \text{artifact}, \text{verification})
\rightarrow
\text{claim boundary}
$$

A public note should make the boundary inspectable without leaking private paths, hostnames, unpublished metrics, or internal task names.

## Scope

- Run records and artifact manifests.
- Checkpoint, config, seed, environment, and data-version boundaries.
- Public-safe module, container, package, and runtime records.
- Reconciliation after interrupted or long-running jobs.
- Public notes that distinguish completed, failed, superseded, and inconclusive runs.

## Notes

- [[infra/reproducibility/run-record|Reproducible run record]]
- [[infra/environments/modules-containers|Environment modules and containers]]

## Checks

- Is the run identity explicit enough to compare two runs?
- Is the artifact type named without publishing a private path?
- Is the split, seed, config, and environment boundary recorded?
- Does the note say whether the run supports a public claim or only a private diagnosis?
- Are failed or interrupted runs represented honestly instead of disappearing?

## Where New Notes Go

- General run records go here.
- Environment capture and module/container notes go here when the point is reproducibility.
- Paper artifact availability goes under [[papers/reproducibility/index|Paper reproducibility]].
- Experiment design and evidence interpretation go under [[concepts/research-methodology/index|Research methodology]].
- Storage-specific artifact problems go under [[infra/io/index|Storage and IO]].

## Related

- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/systems/environment-management|Environment management]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/research-methodology/experiment-ledger|Experiment ledger]]
