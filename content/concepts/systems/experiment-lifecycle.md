---
title: Experiment Lifecycle
tags:
  - systems
  - experiments
  - reproducibility
---

# Experiment Lifecycle

An experiment lifecycle describes how an idea becomes a run, how a run becomes evidence, and how evidence becomes a public claim. It connects [[concepts/research-methodology/experiment-design|Experiment design]], [[concepts/systems/training-run|Training run]], [[concepts/systems/experiment-tracking|Experiment tracking]], and [[concepts/systems/reproducibility|Reproducibility]].

A compact lifecycle is:

$$
q
\rightarrow h
\rightarrow d
\rightarrow r
\rightarrow a
\rightarrow c
$$

where $q$ is a research question, $h$ is a hypothesis, $d$ is an experiment design, $r$ is a run, $a$ is an analysis, and $c$ is a claim.

## Stages

- Question: define what decision the experiment should change.
- Hypothesis: state the expected direction before running.
- Design: choose data, split, baseline, metric, and smallest test.
- Run: execute with fixed code, config, seed policy, data version, and environment.
- Artifact: preserve logs, checkpoints, predictions, figures, and failure notes.
- Analysis: compare against baseline, ablation, confidence interval, or error analysis.
- Claim: write only what the evidence supports.
- Promotion: decide whether the result becomes a note, project milestone, paper reproduction, or Korean post.

## Evidence Flow

$$
\text{claim}
\leftarrow
(\text{question}, \text{hypothesis}, \text{design}, \text{run record}, \text{artifacts}, \text{analysis})
$$

The claim should become weaker when any upstream piece is missing. A result without split details is weaker than a result with a recorded [[concepts/data/dataset-split-contract|Dataset split contract]]. A metric without predictions is weaker than a metric with saved [[concepts/systems/run-artifact|Run artifacts]].

## Checks

- Is the decision question explicit before the run?
- Is the baseline or ablation chosen before seeing the result?
- Are train, validation, and test roles separated?
- Are code commit, config, data version, seed policy, environment, and scheduler assumptions recorded?
- Are artifacts sufficient for later metric checking?
- Does the analysis mention uncertainty, failure modes, or threats to validity?
- Is the public claim separated from private paths, internal task names, unpublished results, and collaborator details?

## Related

- [[concepts/research-methodology/research-question|Research question]]
- [[concepts/research-methodology/hypothesis|Hypothesis]]
- [[concepts/research-methodology/experiment-design|Experiment design]]
- [[concepts/research-methodology/minimum-viable-experiment|Minimum viable experiment]]
- [[concepts/research-methodology/threat-to-validity|Threat to validity]]
- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[papers/reproduction-plan|Reproduction plan]]
- [[infra/reproducible-run-record|Reproducible run record]]
