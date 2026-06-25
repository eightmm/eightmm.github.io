---
title: Decision Record
tags:
  - research
  - projects
  - methodology
---

# Decision Record

A decision record captures why a research or engineering direction changed. It is useful when a project accumulates experiments, paper reviews, and implementation tradeoffs over time.

A decision can be represented as:

$$
D
=
(\text{context}, \text{options}, \text{evidence}, \text{decision}, \text{consequence})
$$

## Suggested Shape

- Context: what decision is being made?
- Options: what alternatives were considered?
- Evidence: papers, experiments, constraints, or failures.
- Decision: what option was chosen?
- Consequence: what becomes easier, harder, or deferred?
- Review trigger: what evidence would cause the decision to change?

## Use Cases

- Choosing a model family for a prototype.
- Choosing a benchmark or split.
- Choosing a data preparation policy.
- Choosing between batch and online inference.
- Deciding not to pursue a research branch.

## Checks

- Is the decision tied to evidence rather than preference?
- Are rejected options recorded fairly?
- Is uncertainty explicit?
- Is the decision public-safe?
- Does it link to the relevant paper, concept, project, or log notes?

## Related

- [[projects/project-milestone-format|Project milestone format]]
- [[concepts/research-methodology/experiment-ledger|Experiment ledger]]
- [[concepts/research-methodology/literature-synthesis|Literature synthesis]]
- [[papers/evidence-table|Evidence table]]
- [[logs/public-log-format|Public log format]]
