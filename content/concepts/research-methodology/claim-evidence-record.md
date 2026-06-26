---
title: Claim Evidence Record
tags:
  - research
  - methodology
  - reproducibility
---

# Claim Evidence Record

A claim evidence record links a public claim to the evidence that supports it. It is the bridge between [[concepts/systems/experiment-lifecycle|Experiment lifecycle]], [[concepts/systems/run-artifact|Run artifact]], [[concepts/research-methodology/result-interpretation|Result interpretation]], and paper-style [[papers/analysis/evidence-table|Evidence table]] writing.

A claim should be represented as:

$$
c
=
(\text{statement}, \text{scope}, \text{evidence}, \text{limitations})
$$

where the statement is the narrow claim, scope states when it applies, evidence lists the supporting runs or sources, and limitations state what the evidence does not prove.

## Evidence Chain

For an experimental claim:

$$
\text{claim}
\leftarrow
(\text{question}, \text{hypothesis}, \text{design}, \text{run}, \text{artifact}, \text{analysis})
$$

For a literature claim:

$$
\text{claim}
\leftarrow
(\text{paper}, \text{figure/table}, \text{task}, \text{dataset}, \text{metric}, \text{baseline})
$$

In both cases, the claim should become weaker when split details, baselines, uncertainty, or artifact availability are missing.

## Record Fields

- Claim: one narrow sentence.
- Scope: dataset, task, model family, metric, split, and deployment assumption.
- Evidence source: run identifier, public artifact, paper figure, table, or proof.
- Data: dataset version, lineage, split, and preprocessing contract.
- Metric: definition, aggregation rule, confidence interval or seed variance when available.
- Baseline: comparison that makes the claim meaningful.
- Failure modes: known errors, hard slices, and negative results.
- Limitations: what the evidence does not establish.
- Public boundary: what was omitted because it is private, unpublished, or unsafe to disclose.

## Claim Strength

A lightweight evidence strength checklist is:

$$
S(c)
=
\mathbf{1}_{\mathrm{data}}
+
\mathbf{1}_{\mathrm{metric}}
+
\mathbf{1}_{\mathrm{baseline}}
+
\mathbf{1}_{\mathrm{uncertainty}}
+
\mathbf{1}_{\mathrm{artifact}}
+
\mathbf{1}_{\mathrm{limitation}}
$$

This is not a real score. It is a reminder that unsupported claims should stay narrow.

## Checks

- Is the claim narrower than the evidence?
- Is the split and data lineage clear enough to interpret generalization?
- Is the metric aligned with the task and decision?
- Is there a baseline, ablation, or previous run for comparison?
- Is uncertainty, seed variance, or confidence interval needed?
- Are missing artifacts marked `not released`, `to verify`, or `not applicable`?
- Does the public record avoid private paths, internal task names, collaborator details, and unpublished results?

## Related

- [[concepts/systems/experiment-lifecycle|Experiment lifecycle]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[concepts/data/data-lineage|Data lineage]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
- [[concepts/research-methodology/threat-to-validity|Threat to validity]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[infra/reproducibility/run-record|Reproducible run record]]
