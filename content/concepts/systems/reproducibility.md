---
title: Reproducibility
tags:
  - systems
  - methodology
  - reproducibility
---

# Reproducibility

Reproducibility means another run can support the same claim under the stated conditions. It is stronger than remembering what was done and weaker than guaranteeing identical bits on every machine.

A reproducible claim should identify:

$$
\text{claim}
\leftarrow
(\text{data}, \text{code}, \text{config}, \text{environment}, \text{protocol}, \text{metric})
$$

## Levels

- Conceptual reproducibility: the idea can be reimplemented and tested.
- Experimental reproducibility: the same code and data produce comparable metrics.
- Operational reproducibility: the workflow can be rerun by another person or agent.
- Bitwise reproducibility: outputs match exactly under controlled settings.

## Checks

- Is the dataset version recoverable?
- Are preprocessing and split construction deterministic?
- Are seeds, hardware assumptions, and nondeterministic kernels documented?
- Are metrics computed by the same script and aggregation rule?
- Are public claims separated from private experiments and unpublished results?

## Related

- [[concepts/data/benchmark|Benchmark]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[infra/reproducible-run-record|Reproducible run record]]
