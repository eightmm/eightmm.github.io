---
title: Paper Review Workflow
tags:
  - papers
  - workflows
---

# Paper Review Workflow

A paper review workflow turns a public paper into a verified note, reusable concept updates, and optional synthesis writing. The goal is not to summarize everything; it is to extract claims that can be connected and checked later.

## Flow

1. Triage the paper into a topic bucket.
2. Verify metadata and source links.
3. Write the paper-specific note using [[papers/paper-note-format|Paper note format]].
4. Extract reusable concepts into [[concepts/index|Concepts]].
5. Link research relevance into [[research/index|Research]].
6. Promote mature themes into Korean [[posts/index|Posts]].

## Evidence Levels

- Metadata verified: title, authors, venue or preprint source, and link are checked.
- Method understood: objective, architecture, data, and evaluation are identified.
- Claims checked: results are tied to metric, split, and benchmark.
- Baseline checked: comparisons and ablations support the claimed contribution.
- Limits recorded: failure modes, assumptions, and missing comparisons are explicit.
- Synthesis ready: the paper can support a public blog post or research map.

## Checks

- Are formulas rewritten with symbol definitions rather than copied blindly?
- Are metrics connected to [[concepts/evaluation/metric|Metric]] and split protocol?
- Are [[concepts/evaluation/baseline|baselines]] and [[concepts/evaluation/ablation-study|ablation studies]] sufficient?
- Are domain risks such as leakage, scaffold split, protein family split, or invalid geometry noted?
- Are uncertain claims marked as unresolved?

## Related

- [[papers/index|Papers]]
- [[papers/reading-status|Reading status]]
- [[agents/paper-brief-workflow|Paper brief workflow]]
- [[agents/human-in-the-loop|Human in the loop]]
