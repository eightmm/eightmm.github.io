---
title: Research Log
tags:
  - research
  - logs
  - methodology
---

# Research Log

A research log records the question, hypothesis, setup, result, interpretation, and next decision. It should preserve the reasoning without exposing private infrastructure, collaborators, or unpublished sensitive results.

Public research logs should be classified with [[logs/public-log-taxonomy|Public log taxonomy]] before promotion into a project, paper, concept, or Korean post.

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
- Does it remain a log, or should it promote through [[logs/log-promotion-rule|Log promotion rule]]?

## Related

- [[logs/public-log-format|Public log format]]
- [[logs/public-log-taxonomy|Public log taxonomy]]
- [[logs/log-promotion-rule|Log promotion rule]]
- [[concepts/research-methodology/experiment-ledger|Experiment ledger]]
- [[concepts/research-methodology/negative-result|Negative result]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[projects/project-milestone-format|Project milestone format]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
