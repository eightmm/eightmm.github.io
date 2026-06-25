---
title: Research Log
tags:
  - research
  - logs
  - methodology
---

# Research Log

A research log records the question, hypothesis, setup, result, interpretation, and next decision. It should preserve the reasoning without exposing private infrastructure, collaborators, or unpublished sensitive results.

## Minimal Shape

```markdown
## Question

What did this run try to answer?

## Hypothesis

What was expected before the run?

## Setup

Dataset, split, baseline, metric, and one changed variable.

## Result

Public-safe result summary.

## Interpretation

Confirmed, not confirmed, surprising, or inconclusive.

## Next

What should change next?
```

## Checks

- Does the log include the baseline and metric?
- Does it record negative or null results?
- Does it avoid exact private paths, account names, server names, internal task names, and unpublished sensitive metrics?
- Can the public lesson be reused without exposing the private run?
- Is it linked to relevant concepts, papers, and project pages?

## Related

- [[logs/public-log-format|Public log format]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[projects/project-milestone-format|Project milestone format]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
