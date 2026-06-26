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

## Reproducibility Tuple

A reproducible result should identify:

$$
\rho
=
(\text{code commit},
\text{diff state},
\text{data version},
\text{split},
\text{config},
\text{seed},
\text{environment},
\text{hardware class},
\text{metric script})
$$

The goal is not to publish private machine details. The goal is to preserve enough public-safe context to know what was tested and what would need to be rerun.

## Determinism Boundary

Exact determinism is often unavailable:

$$
\hat{m}
=
m
\pm
\epsilon_{\mathrm{seed}}
\pm
\epsilon_{\mathrm{system}}
\pm
\epsilon_{\mathrm{data}}
$$

where $m$ is the target metric and the error terms represent seed variation, nondeterministic systems behavior, and data or preprocessing drift. Reporting uncertainty is usually more honest than claiming bitwise reproducibility.

## Failure Modes

- Result cannot be tied to a code commit or data split.
- Preprocessing or metric script changed after model selection.
- Environment was recorded only after the run succeeded.
- Seed, sampler state, or checkpoint state is missing.
- Public summary claims more than the reproducible artifact supports.

## Checks

- Is the dataset version recoverable?
- Are preprocessing and split construction deterministic?
- Are seeds, hardware assumptions, and nondeterministic kernels documented?
- Are run artifacts enough to recompute or inspect the reported metric?
- Are metrics computed by the same script and aggregation rule?
- Are public claims separated from private experiments and unpublished results?
- Is the exact final model-selection rule recorded?
- Are failed or excluded runs accounted for when they affect the claim?
- Can another person identify which artifact produced the reported number?

## Related

- [[concepts/systems/experiment-lifecycle|Experiment lifecycle]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/environment-management|Environment management]]
- [[concepts/data/benchmark|Benchmark]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[infra/reproducible-run-record|Reproducible run record]]
