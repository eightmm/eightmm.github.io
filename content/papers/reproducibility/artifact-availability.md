---
title: Artifact Availability
unlisted: true
aliases:
  - papers/artifact-availability
tags:
  - papers
  - reproducibility
  - artifacts
---

# Artifact Availability

Artifact availability records which public materials are available for checking a paper. It is narrower than full reproducibility: before planning a reproduction, first ask what artifacts exist.

A paper artifact set can be written as:

$$
A(p)
=
\{\text{code}, \text{data}, \text{splits}, \text{config}, \text{weights}, \text{logs}, \text{predictions}, \text{environment}\}
$$

where $p$ is the paper. Missing artifacts should be marked `not found` or `to verify`, not guessed.

## Artifact Types

- Code: repository, license, commit, runnable entry points, scripts.
- Data: public source, license, processed files, filtering policy.
- Splits: train/validation/test split files or deterministic split rule.
- Config: model, optimizer, schedule, seed, precision, and preprocessing settings.
- Weights: checkpoints, model cards, hashes, and loading instructions.
- Logs: training curves, validation records, run metadata, failure notes.
- Predictions: test predictions, generated samples, retrieval outputs, or benchmark submissions.
- Environment: package versions, CUDA/runtime notes, container, or lockfile.

## Artifact Dependency Graph

Artifacts depend on each other:

$$
\text{data}
\rightarrow
\text{preprocessing}
\rightarrow
\text{splits}
\rightarrow
\text{config}
\rightarrow
\text{run}
\rightarrow
\text{predictions}
\rightarrow
\text{metric}
$$

For a benchmark claim, missing split files can be more serious than missing model weights. For an architecture claim, missing training config can be more serious than missing pretrained weights.

## Availability Matrix

| Artifact | Status | Notes |
| --- | --- | --- |
| Code | `to verify` | Repository, commit, license |
| Data | `to verify` | Public source and filtering |
| Splits | `to verify` | Split files or deterministic rule |
| Config | `to verify` | Training and evaluation settings |
| Weights | `to verify` | Checkpoint or model card |
| Logs | `to verify` | Metrics and run metadata |
| Predictions | `to verify` | Outputs for independent scoring |
| Environment | `to verify` | Runtime and dependency contract |

## Claim-Specific Minimums

| Claim | Minimum artifacts |
|---|---|
| architecture improvement | code, config, baseline config, training/eval script |
| benchmark score | data version, split, metric script, predictions or runnable eval |
| generative model | sampler config, sample count, filtering rule, generated samples |
| docking or pose | receptor/ligand preparation, pose files, atom mapping, metric script |
| protein model | sequence/structure source, MSA/template policy, split, residue mapping |
| agent workflow | task suite, tool boundary, verifier, logs or traces |

## Why It Matters

- Code without splits cannot reproduce an evaluation claim.
- Data without preprocessing can silently change the benchmark.
- Weights without config may be unusable or incomparable.
- Predictions can allow metric checking even when training is too expensive.
- Missing artifacts narrow the strength of a paper claim.
- Artifact availability is evidence for [[papers/reproducibility/implementation-readiness|Implementation readiness]], not proof that a reproduction is worth running.

## Checks

- Are artifacts public, versioned, and licensed for reuse?
- Are split files or split rules available?
- Can preprocessing be reconstructed without private scripts?
- Are model weights tied to the exact architecture and config?
- Are benchmark submissions or predictions available for independent metric checks?
- Is compute cost estimated from public information rather than private assumptions?
- Which missing artifact would block a minimum viable reproduction?
- Which artifact is required for the exact claim, rather than the whole paper?
- Are generated samples, failed cases, and filtered outputs available when generation is claimed?

## Related

- [[papers/reproducibility/checklist|Reproducibility checklist]]
- [[papers/reproducibility/implementation-readiness|Implementation readiness]]
- [[papers/reproducibility/reproduction-plan|Reproduction plan]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/systems/environment-management|Environment management]]
- [[infra/reproducibility/run-record|Reproducible run record]]
