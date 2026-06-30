---
title: Paper Note Format
unlisted: true
aliases:
  - papers/paper-note-format
tags:
  - papers
  - methodology
---

# Paper Note Format

A paper note should preserve what the paper claims, what the method changes, how it is evaluated, and how it connects to reusable concepts. It should not become a copied abstract or a raw reading log.

For public paper reviews intended to read like a beginner-friendly blog post, use [[papers/workflows/longform-paper-review-guide|Longform paper review guide]] before drafting.

For papers that combine AI methods, computational biology objects, and mathematical objectives, start from [[papers/workflows/ai-molecular-math-paper-template|AI Computational Biology Math paper template]].

## Suggested Shape

- Citation: verified title, authors, venue or preprint status, and link.
- Status: one of the values in [[papers/workflows/reading-status|Reading status]].
- Question: what problem the paper tries to solve.
- Method: core model, objective, data, and evaluation protocol.
- Results: what is claimed, with metric and benchmark context.
- Claims: extracted statements following [[papers/analysis/claim-extraction|Claim extraction]].
- Evidence: claim-to-result mapping using [[papers/analysis/evidence-table|Evidence table]].
- Benchmark: task, split, metric, and scope using [[papers/analysis/benchmark-card|Benchmark card]] when needed.
- Baselines: what the method is compared against.
- Ablations: which components explain the result.
- Limits: assumptions, failure modes, missing comparisons, or leakage risks, using [[papers/analysis/limitation-taxonomy|Limitation taxonomy]] when useful.
- Artifacts: public code, data, splits, configs, weights, logs, predictions, and environment using [[papers/reproducibility/artifact-availability|Artifact availability]].
- Reproducibility: public code, data, config, and run details following [[papers/reproducibility/checklist|Reproducibility checklist]].
- Reproduction plan: smallest follow-up experiment, if the paper is worth checking directly.
- Connections: links to concepts, research maps, and projects.

## Depth Levels

Not every paper note needs the same length.

| Depth | Target | Required Content |
| --- | --- | --- |
| Seed note | 5-10 min read | citation, question, claim, method, one equation if relevant, evidence table, limits, links |
| Full note | 15-25 min read | seed note plus architecture walkthrough, tensor/object contracts, ablations, benchmark details, later variants, implementation risks |
| Longform review | 25-40 min read | explanatory Korean article or synthesis post; keep the paper note as the canonical reference |

Architecture papers should usually start as seed notes and be promoted to full notes if they anchor a reusable model family.

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
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[papers/workflows/longform-paper-review-guide|Longform paper review guide]]
- [[papers/workflows/reading-status|Reading status]]
- [[concepts/ai-computational-biology-math-contract|AI Computational Biology Math contract]]
- [[papers/workflows/ai-molecular-math-paper-template|AI Computational Biology Math paper template]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/analysis/benchmark-card|Benchmark card]]
- [[papers/analysis/ablation-map|Ablation map]]
- [[papers/analysis/limitation-taxonomy|Limitation taxonomy]]
- [[papers/reproducibility/artifact-availability|Artifact availability]]
- [[papers/reproducibility/checklist|Reproducibility checklist]]
- [[papers/reproducibility/reproduction-plan|Reproduction plan]]
- [[papers/analysis/paper-comparison-matrix|Paper comparison matrix]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
