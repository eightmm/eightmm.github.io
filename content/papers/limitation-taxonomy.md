---
title: Limitation Taxonomy
tags:
  - papers
  - methodology
  - evaluation
---

# Limitation Taxonomy

A limitation taxonomy gives consistent names to the weak points in a paper. It keeps paper notes from saying only "limited evaluation" or "needs more work" without explaining what kind of limitation matters.

## Common Limitation Types

- Data limitation: small dataset, biased sampling, missing labels, noisy labels, or unclear preprocessing.
- Split limitation: random split where scaffold, family, time, source, or entity split is needed.
- Metric limitation: metric does not match the claimed behavior.
- Baseline limitation: missing strong or relevant comparisons.
- Ablation limitation: component claims are not isolated.
- Generalization limitation: claim extends beyond tested distributions.
- Reproducibility limitation: code, data, config, seeds, or compute details are missing.
- Efficiency limitation: memory, latency, throughput, or cost is not measured.
- Domain limitation: chemistry, biology, structure, assay, or operational assumptions are underspecified.
- Interpretation limitation: mechanistic or scientific explanation is broader than the evidence.

## Narrowing a Claim

When a limitation is found, narrow the claim:

$$
\text{broad claim}
\rightarrow
\text{claim under tested data, metric, baseline, and split}
$$

For example, "method improves docking" should become "method improves pose-quality metric on benchmark $B$ under protocol $P$ compared with baseline $C$" if that is what the evidence supports.

## Checks

- Is the limitation about data, method, evaluation, compute, or interpretation?
- Does it weaken the main claim or only a secondary claim?
- Can the paper's claim be narrowed instead of rejected?
- Is the limitation specific enough to guide a follow-up experiment?
- Should a related concept note be updated with this failure mode?

## Related

- [[papers/evidence-table|Evidence table]]
- [[papers/claim-extraction|Claim extraction]]
- [[papers/reproducibility-checklist|Reproducibility checklist]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
