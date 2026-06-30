---
title: Limitation Taxonomy
unlisted: true
aliases:
  - papers/limitation-taxonomy
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

## Severity

Not every limitation blocks the same kind of use.

| Severity | Meaning | Action |
| --- | --- | --- |
| blocking | main claim is not supported | do not cite as evidence for that claim |
| major | claim must be narrowed | record narrower scope |
| moderate | result is useful but incomplete | add follow-up check |
| minor | does not change main interpretation | note briefly |
| unknown | evidence missing | mark `to verify` |

Severity should be tied to the paper's claim, not to personal preference.

## Narrowing a Claim

When a limitation is found, narrow the claim:

$$
\text{broad claim}
\rightarrow
\text{claim under tested data, metric, baseline, and split}
$$

For example, "method improves docking" should become "method improves pose-quality metric on benchmark $B$ under protocol $P$ compared with baseline $C$" if that is what the evidence supports.

## Limitation to Follow-Up

Each important limitation should map to a concrete next check:

| Limitation | Follow-up |
| --- | --- |
| weak split | rerun or inspect grouped split |
| missing baseline | compare with simple strong baseline |
| missing ablation | map component to ablation |
| missing artifact | check code/data/config availability |
| domain ambiguity | update concept note with assumption |

## Checks

- Is the limitation about data, method, evaluation, compute, or interpretation?
- Does it weaken the main claim or only a secondary claim?
- Can the paper's claim be narrowed instead of rejected?
- Is the limitation specific enough to guide a follow-up experiment?
- Should a related concept note be updated with this failure mode?
- Is the limitation tied to a specific claim, table, figure, or protocol?
- Does the note narrow the claim instead of only criticizing broadly?

## Related

- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/reproducibility/checklist|Reproducibility checklist]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/evaluation/claim-evidence-boundary|Claim-evidence boundary]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
