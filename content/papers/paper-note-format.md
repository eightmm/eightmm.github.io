---
title: Paper Note Format
tags:
  - papers
  - methodology
---

# Paper Note Format

A paper note should preserve what the paper claims, what the method changes, how it is evaluated, and how it connects to reusable concepts. It should not become a copied abstract or a raw reading log.

## Suggested Shape

- Citation: verified title, authors, venue or preprint status, and link.
- Question: what problem the paper tries to solve.
- Method: core model, objective, data, and evaluation protocol.
- Results: what is claimed, with metric and benchmark context.
- Limits: assumptions, failure modes, missing comparisons, or leakage risks.
- Connections: links to concepts, research maps, and projects.

## Formula Slot

When a method depends on a mathematical object, include the canonical equation and define symbols. For example, a supervised objective is:

$$
\hat{\theta}
= \arg\min_\theta
\frac{1}{n}\sum_{i=1}^{n}
\mathcal{L}(f_\theta(x_i), y_i)
$$

## Checks

- Are metadata and links verified?
- Are claims separated from personal interpretation?
- Are metrics tied to a split and evaluation protocol?
- Are figures or tables summarized rather than copied?
- Are uncertain details marked as unresolved?

## Related

- [[papers/index|Papers]]
- [[agents/paper-brief-workflow|Paper brief workflow]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
