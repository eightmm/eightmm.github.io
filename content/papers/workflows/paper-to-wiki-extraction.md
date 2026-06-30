---
title: Paper To Wiki Extraction
unlisted: true
tags:
  - papers
  - workflows
  - concepts
---

# Paper To Wiki Extraction

Paper to wiki extraction turns one paper into reusable notes without turning the site into a pile of paper summaries. The paper note keeps source-specific evidence; wiki notes keep definitions, formulas, contracts, and failure modes that future posts can reuse.

$$
\text{paper}
\rightarrow
\text{claims}
\rightarrow
\text{wiki updates}
\rightarrow
\text{post bundle}
$$

## Extraction Passes

| Pass | Extract | Update |
| --- | --- | --- |
| Claim | one sentence claim, supported scope, unsupported scope | [Claim extraction](/papers/analysis/claim-extraction), [Claim-evidence boundary](/concepts/evaluation/claim-evidence-boundary) |
| Object | modeled entity, example unit, output unit, context | [Entities](/entities), [Computational biology entities](/molecular-modeling/entities) |
| Representation | token, graph, coordinate, embedding, conformer, feature cache | [Representation contract](/concepts/modalities/representation-contract) |
| Method | architecture, learning signal, sampler, inference procedure | [AI](/ai), [AI paper claim patterns](/ai/paper-claim-patterns) |
| Formula | objective, likelihood, estimator, metric, update rule | [Formula intake](/math/formula-intake), [Formula pattern catalog](/math/formula-patterns) |
| Data | dataset, preprocessing, split unit, leakage risk, label semantics | [Benchmark intake](/concepts/data/benchmark-intake), [Computational Biology paper intake](/molecular-modeling/paper-intake) |
| Evidence | metric, baseline, ablation, uncertainty, strongest/weakest result | [Evidence table](/papers/analysis/evidence-table), [Benchmark claim contract](/concepts/evaluation/benchmark-claim-contract) |
| Artifact | code, data, split, weight, config, environment, reproducibility status | [Reproducibility checklist](/papers/reproducibility/checklist) |
| Public boundary | private information, unpublished results, collaborator details | [Publishing gate](/inbox/publishing-gate) |
| Post route | reader question, next path, minimum support notes | [Wiki bundle checklist](/posts/workflows/wiki-bundle-checklist) |

## Claim Decomposition

For each important claim, write the local decomposition before editing wiki notes.

```yaml
claim: to verify
paper_scope: to verify
object: to verify
representation: to verify
method: to verify
formula: to verify
data_and_split: to verify
metric: to verify
baseline: to verify
uncertainty: to verify
artifact_status: to verify
wiki_updates:
  - to verify
post_candidate: not applicable
```

## Update Decision

| If The Paper Adds | Prefer |
| --- | --- |
| a reusable term or boundary | concept note update |
| a new object or example unit | entity or data note update |
| a new input construction | representation contract update |
| a new model family or block | architecture note update |
| a new training signal | learning-method or objective note update |
| a new sampler or distribution path | generative model and Math note update |
| a new benchmark protocol | benchmark or evaluation note update |
| only a paper-specific score | paper note only |
| only an unverified candidate | inbox item only |
| several linked reusable updates | Korean synthesis post candidate |

## AI + Computational Biology + Math Pattern

Mixed papers usually need three synchronized updates:

1. Computational Biology: object, label, split, leakage, and deployment context.
2. AI: architecture, learning signal, sampler, inference cost, and evaluation risk.
3. Math: objective, distribution, metric, symbol table, and estimator boundary.

Do not let one axis hide the others. A protein-ligand paper can be a computational biology paper even when it uses a Transformer; a new score or velocity objective still needs Math; a benchmark score still needs an evidence boundary.

## Stop Conditions

- The source is not public.
- The paper metadata is missing and cannot be marked `to verify`.
- The claim cannot be separated from private project context.
- The formula is copied without local symbol definitions.
- The benchmark lacks split, metric, or baseline context.
- The wiki update would only repeat a paper summary.
- The post would have no reusable support notes.

## Related

- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[papers/workflows/claim-routing|Claim routing]]
- [[papers/workflows/concept-update-contract|Concept update contract]]
- [[papers/workflows/ai-molecular-math-readiness-gate|AI Computational Biology Math readiness gate]]
- [[concepts/coverage-matrix|Coverage matrix]]
- [[posts/workflows/wiki-bundle-checklist|Wiki bundle checklist]]
