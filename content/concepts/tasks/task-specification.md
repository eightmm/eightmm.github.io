---
title: Task Specification
tags:
  - tasks
  - evaluation
  - machine-learning
---

# Task Specification

A task specification defines what a model receives, what it must output, what outputs are valid, and how success is measured. It is the contract between data, model, and evaluation.

A compact task specification is:

$$
\mathcal{T}
=
(\mathcal{X}, \mathcal{Y}, v, \mathcal{L}, \mathcal{M}, s)
$$

where $\mathcal{X}$ is the input space, $\mathcal{Y}$ is the [[concepts/tasks/task-output-space|output space]], $v$ is a validity function, $\mathcal{L}$ is the training loss, $\mathcal{M}$ is the metric set, and $s$ is the split rule.

## Required Fields

- Input: raw modality and model-ready representation.
- Output: class, scalar, rank, retrieved set, sequence, mask, box, graph, coordinate object, answer, or action.
- Validity: syntax, geometry, schema, label space, constraints, or safety boundary.
- Loss: what is optimized during training.
- Metric: what is reported during evaluation, selected with [[concepts/evaluation/metric-selection|Metric selection]].
- Split: what generalization claim the result supports.
- Failure mode: invalid output, wrong output, unsupported output, missing evidence, or low-confidence output, using [[concepts/evaluation/failure-mode-taxonomy|Failure mode taxonomy]].

## Task Contract Table

| Field | Question | Example |
| --- | --- | --- |
| example unit | What is one supervised or evaluated case? | molecule-target pair, prompt, image, protein structure |
| input space | What can the model observe? | graph, tokens, coordinates, retrieved context |
| output space | What counts as a prediction? | class, scalar, rank, sequence, pose, mask |
| validity rule | Which outputs are admissible? | valid molecule, normalized probability, no atom clash, JSON schema |
| loss | What signal updates parameters? | cross entropy, MSE, contrastive loss, diffusion loss |
| metric | What claim is reported? | AUROC, RMSE, NDCG, RMSD, enrichment factor |
| split unit | What is held out? | scaffold, protein family, source, time, user |
| aggregation | How are per-example scores summarized? | row mean, target mean, scaffold mean, paired comparison |

## Why It Matters

Two systems can use the same architecture but solve different tasks. A decoder-only transformer can do sequence generation, retrieval reranking, tool-call structured output, or classification depending on the task specification.

The task should also be checked against [[concepts/modalities/modality-task-map|Modality-task map]] so the raw modality, representation, output space, loss, metric, and split rule stay aligned.

## Common Mismatches

| Mismatch | Symptom | Fix |
| --- | --- | --- |
| loss differs from claim | good training loss but weak reported metric | add metric-aligned validation or reranking objective |
| output validity ignored | generated objects are syntactically or physically invalid | define validity before scoring |
| split weaker than claim | random split used for novelty claim | choose split unit that matches deployment |
| example unit unclear | duplicate entities inflate the score | define entity, pair, assay, pose, or region unit |
| metric aggregated by row only | frequent targets or easy cases dominate | report grouped or stratified metrics |
| preprocessing hidden | inference cannot reproduce benchmark input | link a preprocessing contract |

## Computational Biology Examples

| Task | Example Unit | Output Space | Split Unit | Metric Boundary |
| --- | --- | --- | --- | --- |
| molecular property prediction | molecule or assay row | scalar or class | scaffold, source, assay | unit, threshold, endpoint must be explicit |
| activity prediction | molecule-target-assay tuple | scalar, class, or ranking score | scaffold and target family | assay harmonization affects comparability |
| docking pose ranking | receptor-ligand pose | pose score or rank | complex, protein family, ligand scaffold | pose RMSD and physical validity are separate |
| protein sequence modeling | sequence or residue | token, embedding, annotation, scalar | sequence identity cluster or family | homolog leakage changes interpretation |
| variant effect prediction | variant or region | scalar, class, or rank | locus, gene, chromosome, source | reference genome and coordinate convention matter |

## Checks

- What is one example and one target?
- Is the output independent, ranked, sequential, spatial, temporal, or structured?
- Are invalid outputs counted as failures or repaired before scoring?
- Does the metric measure the actual downstream behavior?
- Does the split unit match the claimed generalization?
- Is the training loss aligned with the evaluation metric?

## Related

- [[concepts/tasks/index|Tasks]]
- [[concepts/tasks/task-output-space|Task output space]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/failure-mode-taxonomy|Failure mode taxonomy]]
- [[concepts/modalities/index|Modalities]]
