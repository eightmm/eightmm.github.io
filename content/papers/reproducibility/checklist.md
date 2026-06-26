---
title: Reproducibility Checklist
unlisted: true
aliases:
  - papers/reproducibility-checklist
tags:
  - papers
  - reproducibility
  - methodology
---

# Reproducibility Checklist

A reproducibility checklist captures whether a paper gives enough information for another researcher to rerun, reimplement, or fairly compare the method.

Reproducibility is not binary. A paper can be reproducible at different levels:

$$
\text{paper}
\rightarrow
\{\text{metadata}, \text{data}, \text{code}, \text{config}, \text{compute}, \text{evaluation}\}
$$

## Checklist

- Metadata: title, authors, venue or preprint source, and link are verified.
- Code: repository, license, commit, and runnable entry points are identified when public.
- Data: dataset version, filtering, preprocessing, splits, and licenses are described.
- Model: architecture, objective, initialization, and hyperparameters are specified.
- Training: optimizer, schedule, batch size, seeds, precision, hardware, and stopping rule are specified.
- Evaluation: metric, benchmark, split, baseline, and aggregation are specified.
- Compute: resource scale is described enough to estimate feasibility.
- Artifacts: checkpoints, logs, predictions, or processed datasets are available when claimed.

## AI and Computational Biology Additions

| Area | Extra Reproducibility Field |
|---|---|
| molecule model | standardization, chemical state, scaffold split, featurizer version |
| protein model | sequence source, MSA/template policy, residue mapping, family split |
| docking or pose | receptor/ligand preparation, atom mapping, pose validity, failed docking count |
| generative model | sampler config, sample count, filtering, invalid denominator |
| probability model | calibration data, threshold rule, probability metric |
| agent workflow | task suite, tool versions, verifier, logs/traces, human boundary |

## Minimum Reimplementation Record

For a method claim, the minimum record is:

$$
R
=
\{\text{code}, \text{data}, \text{config}, \text{seed}, \text{environment}, \text{metric}\}
$$

If any element is missing, mark it as `to verify` rather than guessing.

For generated outputs or benchmark submissions, include the denominator:

$$
R_{\mathrm{eval}}
=
\{\text{attempted},\text{failed},\text{filtered},\text{scored}\}
$$

Without this, reported quality may describe only retained outputs.

## Checks

- Can the result be rerun from public materials?
- If not, can the method still be reimplemented from the description?
- Are preprocessing and split details complete?
- Are random seeds and run variance reported?
- Is compute cost part of the claim?
- Are missing details severe enough to weaken the paper's conclusions?
- Is the paper [[papers/reproducibility/implementation-readiness|implementation-ready]] for one scoped claim?
- Are public artifacts enough to check the narrow claim, even if full training is impossible?
- Are invalid, failed, or filtered outputs recorded?

## Related

- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[papers/reproducibility/artifact-availability|Artifact availability]]
- [[papers/reproducibility/implementation-readiness|Implementation readiness]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/systems/training-run|Training run]]
- [[concepts/data/data-versioning|Data versioning]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[infra/reproducibility/run-record|Reproducible run record]]
