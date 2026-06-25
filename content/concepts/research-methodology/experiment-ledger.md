---
title: Experiment Ledger
tags:
  - research
  - experiments
  - logs
---

# Experiment Ledger

An experiment ledger is a structured list of experiments, their hypotheses, setup, outcomes, and next decisions. It prevents research from becoming a pile of disconnected runs.

Each experiment can be represented as:

$$
e_i
=
(\mathcal{Q}_i, h_i, c_i, r_i, d_i)
$$

where $\mathcal{Q}_i$ is the question, $h_i$ is the hypothesis, $c_i$ is the configuration, $r_i$ is the result summary, and $d_i$ is the decision.

## Minimal Columns

| Column | Meaning |
| --- | --- |
| ID | Public-safe identifier or short slug. |
| Question | What uncertainty does this reduce? |
| Hypothesis | What did you expect before running? |
| Change | What variable changed from the baseline? |
| Data/Split | Dataset version and evaluation boundary. |
| Metric | Primary metric or diagnostic. |
| Result | Public-safe outcome summary. |
| Interpretation | Confirmed, not confirmed, surprising, or inconclusive. |
| Next | What changes because of this result? |

## Public-Safe Rule

The public ledger should generalize or omit private run identifiers, internal task names, private paths, exact server details, collaborator details, and unpublished sensitive metrics.

## Checks

- Does every experiment have a pre-run hypothesis?
- Is the baseline explicit?
- Is the changed variable isolated?
- Are negative and inconclusive runs recorded?
- Is the next decision clear?

## Related

- [[concepts/research-methodology/research-log|Research log]]
- [[concepts/research-methodology/experiment-design|Experiment design]]
- [[concepts/research-methodology/negative-result|Negative result]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[infra/reproducible-run-record|Reproducible run record]]
