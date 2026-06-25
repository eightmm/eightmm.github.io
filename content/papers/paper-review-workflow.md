---
title: Paper Review Workflow
tags:
  - papers
  - workflows
---

# Paper Review Workflow

A paper review workflow turns a public paper into a verified note, reusable concept updates, and optional synthesis writing. The goal is not to summarize everything; it is to extract claims that can be connected and checked later.

## Flow

1. Triage the paper with [[papers/paper-triage|Paper triage]].
2. Verify metadata and source links.
3. Write the paper-specific note using [[papers/paper-note-format|Paper note format]].
4. Extract claims using [[papers/claim-extraction|Claim extraction]].
5. Map claims to evidence using [[papers/evidence-table|Evidence table]].
6. Create a [[papers/benchmark-card|Benchmark card]] for the benchmark when the evaluation protocol matters.
7. Map component claims with [[papers/ablation-map|Ablation map]] when the paper argues why a method works.
8. Record limits with [[papers/limitation-taxonomy|Limitation taxonomy]].
9. Check reproducibility using [[papers/reproducibility-checklist|Reproducibility checklist]].
10. Write a [[papers/reproduction-plan|Reproduction plan]] only if the paper is worth rerunning or reimplementing.
11. Compare related papers with [[papers/paper-comparison-matrix|Paper comparison matrix]] when useful.
12. Extract reusable concepts into [[concepts/index|Concepts]].
13. Link research relevance into [[research/index|Research]].
14. Synthesize related work with [[concepts/research-methodology/literature-synthesis|Literature synthesis]] when a topic has enough papers.
15. Promote mature themes into Korean [[posts/index|Posts]].

## Evidence Levels

- Metadata verified: title, authors, venue or preprint source, and link are checked.
- Method understood: objective, architecture, data, and evaluation are identified.
- Claims checked: results are tied to metric, split, benchmark, and uncertainty.
- Evidence table complete: main claims have supporting experiments and limits.
- Benchmark card complete: benchmark scope, split, metric, and leakage risks are explicit.
- Ablation map complete: component claims are tied to isolated experiments.
- Baseline checked: comparisons and ablations support the claimed contribution.
- Limits recorded: failure modes, assumptions, and missing comparisons are explicit.
- Reproducibility checked: code, data, config, compute, and evaluation details are marked present or missing.
- Synthesis ready: the paper can support a public blog post or research map.

## Checks

- Is the paper worth a curated note, or should it only update an existing concept?
- Are formulas rewritten with symbol definitions rather than copied blindly?
- Are metrics connected to [[concepts/evaluation/metric|Metric]] and split protocol?
- Are reported gains larger than [[concepts/evaluation/confidence-interval|confidence intervals]] or run-to-run variance?
- Does the paper require [[concepts/evaluation/statistical-significance|statistical significance]] checks?
- Are [[concepts/evaluation/baseline|baselines]] and [[concepts/evaluation/ablation-study|ablation studies]] sufficient?
- Are domain risks such as leakage, scaffold split, protein family split, or invalid geometry noted?
- Are uncertain claims marked as unresolved?

## Related

- [[papers/index|Papers]]
- [[papers/paper-triage|Paper triage]]
- [[papers/reading-status|Reading status]]
- [[papers/claim-extraction|Claim extraction]]
- [[papers/evidence-table|Evidence table]]
- [[papers/benchmark-card|Benchmark card]]
- [[papers/ablation-map|Ablation map]]
- [[papers/limitation-taxonomy|Limitation taxonomy]]
- [[papers/reproducibility-checklist|Reproducibility checklist]]
- [[papers/reproduction-plan|Reproduction plan]]
- [[papers/paper-comparison-matrix|Paper comparison matrix]]
- [[concepts/research-methodology/research-question|Research question]]
- [[concepts/research-methodology/literature-synthesis|Literature synthesis]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
