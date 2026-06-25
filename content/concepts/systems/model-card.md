---
title: Model Card
tags:
  - systems
  - evaluation
  - documentation
---

# Model Card

A model card is a public-facing record of what a model is, what it was trained or evaluated for, what it should not be used for, and what evidence supports those boundaries. It is not a marketing page; it is a compact operating contract for readers and users.

A minimal model card can be represented as:

$$
C
=
(\text{model}, \text{task}, \text{data}, \text{metrics}, \text{limits}, \text{intended use}, \text{risks})
$$

## Required Sections

- Model identity: model family, version, checkpoint, and release status.
- Task: [[concepts/tasks/task-specification|Task specification]], output space, validity rule, and failure modes.
- Input and output: [[concepts/systems/inference-contract|Inference contract]] for accepted requests and returned results.
- Data: training/evaluation data sources, preprocessing, split rule, and known exclusions.
- Evaluation: primary metric, diagnostic metrics, confidence intervals when available, and failure analysis.
- Intended use: what decisions or workflows the model can support.
- Out-of-scope use: tasks, domains, or inputs where the model should not be trusted.
- Limitations: uncertainty, OOD behavior, bias, missing modalities, leakage risks, or reproducibility gaps.
- Operational notes: latency, throughput, memory, hardware, environment, and versioning constraints when relevant.

## Evaluation Block

The evaluation block should connect:

$$
(\mathcal{T}, \mathcal{D}_{\mathrm{eval}}, s, M, z)
$$

where $\mathcal{T}$ is the task, $\mathcal{D}_{\mathrm{eval}}$ is the evaluation dataset, $s$ is the split rule, $M$ is the metric set, and $z$ is the failure-mode taxonomy.

## Public Boundary

A public model card should not expose private paths, unreleased checkpoint locations, private datasets, collaborator details, internal task names, or unpublished sensitive results. Missing evidence should be marked `to verify`, `not released`, or `not applicable`.

## Checks

- Is the task and output space explicit?
- Is intended use narrower than "general purpose"?
- Are unsupported inputs and out-of-scope domains named?
- Are metrics tied to a split and evaluation protocol?
- Are failure modes named rather than hidden behind one score?
- Is the inference input/output contract clear enough for another workflow to call safely?
- Are model version, preprocessing version, and environment assumptions recorded?

## Related

- [[concepts/systems/inference-contract|Inference contract]]
- [[concepts/systems/inference|Inference]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/failure-mode-taxonomy|Failure mode taxonomy]]
- [[concepts/data/dataset-card|Dataset card]]
- [[projects/project-note-format|Project note format]]
