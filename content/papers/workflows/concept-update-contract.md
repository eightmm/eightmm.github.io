---
title: Concept Update Contract
unlisted: true
tags:
  - papers
  - workflows
  - concepts
---

# Concept Update Contract

Concept update contract decides what should be extracted from a paper into reusable wiki notes. A paper note records one source. A concept note records reusable structure that future papers and posts can link to.

$$
\text{concept update}
=
\text{definition}
+ \text{formula}
+ \text{contract}
+ \text{evidence boundary}
$$

## Extract When

| Paper Content | Update |
| --- | --- |
| New or recurring entity, modality, or representation | entity, modality, or representation contract note |
| New architecture block or inductive bias | architecture concept note |
| New objective, estimator, or training signal | learning method, machine learning, or Math note |
| New sampling or generation process | generative model note |
| New molecular modeling object boundary | Molecular Modeling or SBDD concept note |
| New benchmark, split, metric, or failure mode | evaluation or data note |
| New reproducibility or artifact lesson | systems, papers/reproducibility, or infra note |
| New agent workflow or verifier behavior | agent note |

## Keep In Paper Note

| Paper-Specific Detail | Reason |
| --- | --- |
| exact result number | belongs to one benchmark table |
| author claim wording | should not become established fact |
| ablation table detail | evidence for the paper's mechanism claim |
| public artifact status | can change and belongs to the paper record |
| unresolved concern | keep as `to verify` or limitation |
| paper-specific comparison | use evidence table or comparison matrix |

## Concept Update Minimum

| Field | Requirement |
| --- | --- |
| Definition | reusable definition independent of one paper |
| Formula | canonical equation or operational rule when useful |
| Scope | when the concept applies and when it does not |
| Contract | object, representation, objective, metric, split, or artifact fields |
| Evidence boundary | what paper evidence supports and what remains unproven |
| Related papers | link paper notes without copying their full summaries |
| Related concepts | connect AI, Computational Biology, Math, Data, Evaluation, Systems, or Agents |

## Update Pattern

When a paper adds a reusable idea, update the concept note with a small section:

```markdown
## Paper Notes

| Paper | Adds | Boundary |
| --- | --- | --- |
| [Paper title](/papers) | reusable idea | claim remains benchmark-specific |
```

Use this only when the paper actually changes how the concept should be understood. Do not turn every paper into a concept-page citation list.

## Anti-Patterns

- Creating a new concept page for every paper acronym.
- Copying a paper summary into a concept note.
- Treating one benchmark result as a general property of a method.
- Adding a formula without symbol definitions.
- Updating only paper notes while leaving the reusable wiki graph unchanged.
- Adding paper-specific private context, unpublished results, or internal project details.

## Workflow

1. Run [[papers/workflows/claim-routing|Claim routing]].
2. Decompose the claim with [[papers/workflows/paper-to-wiki-extraction|Paper to wiki extraction]].
3. Check missing support notes with [[concepts/coverage-matrix|Coverage matrix]].
4. Decide which facts remain paper-specific.
5. Update the smallest useful concept page.
6. Link the concept update back to the paper note or paper-analysis note.
7. If several concept notes now form a reader-facing path, consider promoting them through the Posts workflow.

## Related

- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[papers/workflows/claim-routing|Claim routing]]
- [[papers/workflows/paper-to-wiki-extraction|Paper to wiki extraction]]
- [[concepts/coverage-matrix|Coverage matrix]]
- [[concepts/evaluation/claim-evidence-boundary|Claim-evidence boundary]]
- [[math/formula-explanation-ladder|Formula explanation ladder]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
