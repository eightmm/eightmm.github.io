---
title: Artifact Availability
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

## Why It Matters

- Code without splits cannot reproduce an evaluation claim.
- Data without preprocessing can silently change the benchmark.
- Weights without config may be unusable or incomparable.
- Predictions can allow metric checking even when training is too expensive.
- Missing artifacts narrow the strength of a paper claim.

## Checks

- Are artifacts public, versioned, and licensed for reuse?
- Are split files or split rules available?
- Can preprocessing be reconstructed without private scripts?
- Are model weights tied to the exact architecture and config?
- Are benchmark submissions or predictions available for independent metric checks?
- Is compute cost estimated from public information rather than private assumptions?

## Related

- [[papers/reproducibility-checklist|Reproducibility checklist]]
- [[papers/reproduction-plan|Reproduction plan]]
- [[papers/paper-review-workflow|Paper review workflow]]
- [[papers/evidence-table|Evidence table]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/systems/environment-management|Environment management]]
- [[infra/reproducible-run-record|Reproducible run record]]
