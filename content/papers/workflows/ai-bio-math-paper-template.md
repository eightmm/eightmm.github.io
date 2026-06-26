---
title: AI-Molecular-Math Paper Template
unlisted: true
tags:
  - papers
  - workflows
  - ai
  - molecular-modeling
  - math
---

# AI-Molecular-Math Paper Template

Use this template when a paper connects AI methods, molecular modeling objects, and mathematical objectives. Keep missing metadata as `to verify`; do not invent authors, metrics, artifacts, or benchmark details.

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
- Main axis: AI method / molecular modeling object / Math objective / benchmark / system / agent workflow
- Secondary axes: `to verify`
- Public source: `to verify`
- Reading status: [[papers/workflows/reading-status|Reading status]]

## Routing

| Route | Fill |
| --- | --- |
| AI intake | [AI paper intake](/ai/paper-intake) |
| Molecular modeling intake | [Molecular modeling paper intake](/molecular-modeling/paper-intake) or `not applicable` |
| Formula intake | [Formula intake](/math/formula-intake) |
| Formula explanation | [Formula explanation ladder](/math/formula-explanation-ladder) |
| Benchmark intake | [Benchmark intake](/concepts/data/benchmark-intake) |
| Claim routing | [Claim routing](/papers/workflows/claim-routing) |
| Readiness gate | [AI-Molecular-Math readiness gate](/papers/workflows/ai-molecular-math-readiness-gate) |
| Coverage check | [Coverage matrix](/concepts/coverage-matrix) |

## Claim

State claims as paper claims, not facts.

| Claim | Evidence | Status |
| --- | --- | --- |
| `to verify` | figure/table/section `to verify` | to verify |

## Method

- Input object: `to verify`
- Output object: `to verify`
- Representation: `to verify`
- Architecture: `to verify`
- Learning signal: `to verify`
- Objective: `to verify`
- Inference or sampling procedure: `to verify`

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

## Molecular Modeling Contract

Use `not applicable` only when the paper is not about molecular modeling.

| Field | Value |
| --- | --- |
| Biological object | molecule / ligand / protein / pocket / complex / assay / genome region / not applicable |
| Example unit | `to verify` |
| Label semantics | endpoint, unit, direction, threshold, censoring, source, or `not applicable` |
| Preprocessing | molecule standardization, protein cleaning, coordinate source, or `not applicable` |
| Split unit | scaffold, protein family, complex pair, assay/source, time, template-aware, or `to verify` |
| Leakage risks | duplicate, scaffold, homolog, template, assay/source, coordinate-frame, or `to verify` |

## Benchmark

| Field | Value |
| --- | --- |
| Dataset or benchmark | `to verify` |
| Task | `to verify` |
| Split | `to verify` |
| Metric | `to verify` |
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
- Evaluation or benchmark note to update: `to verify`
- Potential Korean post route: [[posts/ai-bio-math-post-intake|AI-Molecular-Math post intake]]
- Promotion readiness: [[papers/workflows/ai-molecular-math-readiness-gate|AI-Molecular-Math readiness gate]]

## Related

- [[papers/workflows/paper-note-format|Paper note format]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[papers/workflows/ai-molecular-math-readiness-gate|AI-Molecular-Math readiness gate]]
- [[math/formula-explanation-ladder|Formula explanation ladder]]
- [[concepts/coverage-matrix|Coverage matrix]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/analysis/benchmark-card|Benchmark card]]
