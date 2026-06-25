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
- Status: one of the values in [[papers/reading-status|Reading status]].
- Question: what problem the paper tries to solve.
- Method: core model, objective, data, and evaluation protocol.
- Results: what is claimed, with metric and benchmark context.
- Claims: extracted statements following [[papers/claim-extraction|Claim extraction]].
- Evidence: claim-to-result mapping using [[papers/evidence-table|Evidence table]].
- Baselines: what the method is compared against.
- Ablations: which components explain the result.
- Limits: assumptions, failure modes, missing comparisons, or leakage risks, using [[papers/limitation-taxonomy|Limitation taxonomy]] when useful.
- Reproducibility: public code, data, config, and run details following [[papers/reproducibility-checklist|Reproducibility checklist]].
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
- Are baselines and ablations strong enough to support the claim?
- For chem-bio papers, are assay harmonization, negative sets, activity cliffs, and applicability domain addressed?
- Are figures or tables summarized rather than copied?
- Are uncertain details marked as unresolved?

## Related

- [[papers/index|Papers]]
- [[papers/paper-review-workflow|Paper review workflow]]
- [[papers/reading-status|Reading status]]
- [[papers/claim-extraction|Claim extraction]]
- [[papers/evidence-table|Evidence table]]
- [[papers/limitation-taxonomy|Limitation taxonomy]]
- [[papers/reproducibility-checklist|Reproducibility checklist]]
- [[papers/paper-comparison-matrix|Paper comparison matrix]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
