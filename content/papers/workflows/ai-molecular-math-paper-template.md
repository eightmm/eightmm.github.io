---
title: AI Computational Biology Math Paper Template
aliases:
  - papers/workflows/ai-bio-math-paper-template
  - papers/workflows/ai-computational-biology-math-paper-template
unlisted: true
tags:
  - papers
  - workflows
  - ai
  - molecular-modeling
  - math
---

# AI Computational Biology Math Paper Template

Use this template when a paper connects AI methods, computational biology objects, and mathematical objectives. Keep missing metadata as `to verify`; do not invent authors, metrics, artifacts, or benchmark details.

## Frontmatter

```markdown
---
title: Paper Title
status: reading
paper:
  title: to verify
  authors: to verify
  venue: to verify
  year: to verify
  url: to verify
tags:
  - papers
---
```

## Summary

- One-line question: `to verify`
- Main contribution: `to verify`
- Main axis: AI method / computational biology object / Math objective / benchmark / system / agent workflow
- Secondary axes: `to verify`
- Public source: `to verify`
- Reading status: [[papers/workflows/reading-status|Reading status]]

## Routing

| Route | Fill |
| --- | --- |
| AI intake | [AI paper intake](/ai/paper-intake) |
| Computational biology intake | [Computational Biology paper intake](/molecular-modeling/paper-intake) or `not applicable` |
| Formula intake | [Formula intake](/math/formula-intake) |
| Formula explanation | [Formula explanation ladder](/math/formula-explanation-ladder) |
| Benchmark intake | [Benchmark intake](/concepts/data/benchmark-intake) |
| Claim routing | [Claim routing](/papers/workflows/claim-routing) |
| Cross-axis contract | [AI Computational Biology Math contract](/concepts/ai-computational-biology-math-contract) |
| Readiness gate | [AI Computational Biology Math readiness gate](/papers/workflows/ai-molecular-math-readiness-gate) |
| Coverage check | [Coverage matrix](/concepts/coverage-matrix) |

## Claim

State claims as paper claims, not facts.

| Claim | Evidence | Status |
| --- | --- | --- |
| `to verify` | figure/table/section `to verify` | to verify |

## Claim-Type Gate

Select all that apply before summarizing the result.

| Claim Type | Applies? | Required Check |
| --- | --- | --- |
| architecture improvement | to verify | fair baseline, ablation, parameter/compute boundary |
| SSL or pretraining | to verify | pretraining unit, positive/corruption rule, transfer split |
| generative model | to verify | sampler budget, validity/diversity/novelty/utility, invalid denominator |
| energy or score model | to verify | energy/score/force definition, sampler or negative process, validity after filtering |
| constrained generation | to verify | hard constraint vs penalty/projection/repair/filter, invalid-output denominator |
| property or activity prediction | to verify | label semantics, target/assay context, split unit |
| protein modeling | to verify | sequence/structure source, PLM/MSA/template policy, family split |
| docking or pose | to verify | receptor/ligand preparation, pose metric, failed docking handling |
| benchmark result | to verify | allowed information, selection rule, baseline, uncertainty |
| formula or estimator | to verify | variables, distributions, objective, estimator assumptions |
| agent or tool workflow | to verify | tool boundary, verifier, memory, success/failure definition |

Evidence level:

```yaml
evidence_level: mentioned # mentioned | specified | supported | bounded | reproducible
```

## Method

- Input object: `to verify`
- Output object: `to verify`
- Representation: `to verify`
- Architecture: `to verify`
- Learning signal: `to verify`
- Objective: `to verify`
- Inference or sampling procedure: `to verify`
- Constraint or filtering rule: `to verify`

## Representation Contract

| Field | Value |
| --- | --- |
| Raw unit | molecule / protein / residue / atom / complex / text / image / graph / assay row / to verify |
| Model input | token / graph / coordinate / embedding / pair feature / latent / to verify |
| Tokenization or graph construction | `to verify` |
| Positional or geometric encoding | `to verify` |
| Context source | target / pocket / assay / retrieval / template / prompt / not applicable |
| Information available at inference | `to verify` |
| Information that must be excluded | labels, future data, ground-truth structures, private context, or `to verify` |

## Formula Slot

Rewrite the key equation with symbol definitions. Use level 3 from [[math/formula-explanation-ladder|Formula explanation ladder]] when the paper's claim depends on the equation.

$$
\mathcal{J}(\theta)
=
\mathbb{E}_{u\sim q(u)}
\left[
\ell_\theta(u)
\right]
$$

- $u$: `to verify`
- $q(u)$: `to verify`
- $\ell_\theta(u)$: `to verify`
- $\theta$: `to verify`
- Explanation level: name / canonical formula / operational form / claim contract / derivation
- Relation to evaluation metric: `to verify`

## Objective and Metric Alignment

| Field | Value |
| --- | --- |
| Training objective | `to verify` |
| Objective family | likelihood / contrastive / score / energy / velocity / reward / constraint / metric / to verify |
| Model-selection metric | `to verify` |
| Reported metric | `to verify` |
| Claimed utility | `to verify` |
| Mismatch risk | `to verify` |

## Computational Biology Contract

Use `not applicable` only when the paper is not about computational biology.

| Field | Value |
| --- | --- |
| Biological object | molecule / ligand / protein / pocket / complex / assay / genome region / not applicable |
| Example unit | `to verify` |
| Label semantics | endpoint, unit, direction, threshold, censoring, source, or `not applicable` |
| Preprocessing | molecule standardization, protein cleaning, coordinate source, or `not applicable` |
| Split unit | scaffold, protein family, complex pair, assay/source, time, template-aware, or `to verify` |
| Leakage risks | duplicate, scaffold, homolog, template, assay/source, coordinate-frame, or `to verify` |

## Public Boundary

| Field | Value |
| --- | --- |
| Uses only public sources | to verify |
| No private paths, servers, accounts, ports, or credentials | yes |
| No collaborator-private details | yes |
| No unpublished internal results | yes |
| Sensitive details removed or generalized | to verify |

## Benchmark

| Field | Value |
| --- | --- |
| Dataset or benchmark | `to verify` |
| Task | `to verify` |
| Split | `to verify` |
| Metric | `to verify` |
| Metric family | classification / regression / ranking / probability / generation / pose / enrichment / to verify |
| Baseline | `to verify` |
| Model-selection rule | `to verify` |
| Uncertainty or seed variation | `to verify` |
| Allowed information | training data, templates, retrieval corpus, prompt examples, or `to verify` |

## Evidence and Limits

- Strongest evidence: `to verify`
- Weakest evidence: `to verify`
- Missing baseline or ablation: `to verify`
- Failure modes: `to verify`
- Applicability domain: `to verify`
- Public artifact availability: [[papers/reproducibility/artifact-availability|Artifact availability]]
- Reproducibility status: [[papers/reproducibility/checklist|Reproducibility checklist]]

## Concept Updates

- Object/entity note to update: `to verify`
- Modality/task note to update: `to verify`
- Architecture or learning note to update: `to verify`
- Math note to update: `to verify`
- Constraint/objective note to update: [[concepts/math/constrained-optimization|Constrained optimization]], [[concepts/generative-models/energy-based-model|Energy-based model]], or `not applicable`
- Evaluation or benchmark note to update: `to verify`
- Potential Korean post route: [[posts/ai-molecular-math-post-intake|AI computational biology math post intake]]
- Promotion readiness: [[papers/workflows/ai-molecular-math-readiness-gate|AI computational biology math readiness gate]]

## Related

- [[papers/workflows/paper-note-format|Paper note format]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[concepts/ai-computational-biology-math-contract|AI Computational Biology Math contract]]
- [[papers/workflows/ai-molecular-math-readiness-gate|AI computational biology math readiness gate]]
- [[math/formula-explanation-ladder|Formula explanation ladder]]
- [[concepts/coverage-matrix|Coverage matrix]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/analysis/benchmark-card|Benchmark card]]
