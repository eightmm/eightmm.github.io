---
title: Paper Review Workflow
unlisted: true
aliases:
  - papers/paper-review-workflow
tags:
  - papers
  - workflows
---

# Paper Review Workflow

A paper review workflow turns a public paper into a verified note, reusable concept updates, and optional synthesis writing. The goal is not to summarize everything; it is to extract claims that can be connected and checked later.

## Flow

1. Triage the paper with [[papers/workflows/paper-triage|Paper triage]].
2. Verify metadata and source links.
3. Route the paper through [[ai/paper-intake|AI paper intake]], [[bio/paper-intake|Molecular modeling paper intake]], or [[math/formula-intake|Formula intake]] as needed.
4. Check missing support notes with [[concepts/coverage-matrix|Coverage matrix]].
5. Write the paper-specific note using [[papers/workflows/paper-note-format|Paper note format]] or [[papers/workflows/ai-bio-math-paper-template|AI-Molecular-Math paper template]].
6. Extract claims using [[papers/analysis/claim-extraction|Claim extraction]].
7. Map claims to evidence using [[papers/analysis/evidence-table|Evidence table]].
8. Create a [[papers/analysis/benchmark-card|Benchmark card]] for the benchmark when the evaluation protocol matters.
9. Map component claims with [[papers/analysis/ablation-map|Ablation map]] when the paper argues why a method works.
10. Record limits with [[papers/analysis/limitation-taxonomy|Limitation taxonomy]].
11. Record public artifacts using [[papers/reproducibility/artifact-availability|Artifact availability]].
12. Check reproducibility using [[papers/reproducibility/checklist|Reproducibility checklist]].
13. Decide [[papers/reproducibility/implementation-readiness|Implementation readiness]] before spending compute.
14. Write a [[papers/reproducibility/reproduction-plan|Reproduction plan]] only if the paper is worth rerunning or reimplementing.
15. Record any rerun, reimplementation, or diagnostic as [[papers/reproducibility/reproduction-result|Reproduction result]].
16. Compare related papers with [[papers/analysis/paper-comparison-matrix|Paper comparison matrix]] when useful.
17. Extract reusable concepts into [[concepts/index|Concepts]].
18. Link research relevance into [[research/index|Research]].
19. Synthesize related work with [[concepts/research-methodology/literature-synthesis|Literature synthesis]] when a topic has enough papers.
20. Promote mature themes into Korean [[posts/index|Posts]].

## Evidence Levels

- Metadata verified: title, authors, venue or preprint source, and link are checked.
- Method understood: objective, architecture, data, and evaluation are identified.
- Claims checked: results are tied to metric, split, benchmark, and uncertainty.
- Evidence table complete: main claims have supporting experiments and limits.
- Benchmark card complete: benchmark scope, split, metric, and leakage risks are explicit.
- Ablation map complete: component claims are tied to isolated experiments.
- Baseline checked: comparisons and ablations support the claimed contribution.
- Limits recorded: failure modes, assumptions, and missing comparisons are explicit.
- Artifact availability checked: public code, data, splits, configs, weights, logs, predictions, and environment are marked present or missing.
- Reproducibility checked: code, data, config, compute, and evaluation details are marked present or missing.
- Implementation readiness checked: the target claim, minimum viable experiment, compute estimate, and failure interpretation are explicit.
- Reproduction result recorded: the rerun or diagnostic outcome is tied to artifacts, metrics, limitations, and next decision.
- Synthesis ready: the paper can support a public blog post or research map.

## Checks

- Is the paper worth a curated note, or should it only update an existing concept?
- Are formulas rewritten with symbol definitions rather than copied blindly?
- Has the paper been routed through the relevant intake page: [[ai/paper-intake|AI]], [[bio/paper-intake|Molecular Modeling]], or [[math/formula-intake|Math]]?
- Are metrics connected to [[concepts/evaluation/metric|Metric]] and split protocol?
- Are reported gains larger than [[concepts/evaluation/confidence-interval|confidence intervals]] or run-to-run variance?
- Does the paper require [[concepts/evaluation/statistical-significance|statistical significance]] checks?
- Are [[concepts/evaluation/baseline|baselines]] and [[concepts/evaluation/ablation-study|ablation studies]] sufficient?
- Are domain risks such as leakage, scaffold split, protein family split, or invalid geometry noted?
- Are uncertain claims marked as unresolved?

## Related

- [[papers/index|Papers]]
- [[papers/workflows/paper-triage|Paper triage]]
- [[papers/workflows/reading-status|Reading status]]
- [[papers/workflows/ai-bio-math-paper-template|AI-Molecular-Math paper template]]
- [[ai/paper-intake|AI paper intake]]
- [[bio/paper-intake|Molecular modeling paper intake]]
- [[math/formula-intake|Formula intake]]
- [[concepts/coverage-matrix|Coverage matrix]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/analysis/benchmark-card|Benchmark card]]
- [[papers/analysis/ablation-map|Ablation map]]
- [[papers/analysis/limitation-taxonomy|Limitation taxonomy]]
- [[papers/reproducibility/artifact-availability|Artifact availability]]
- [[papers/reproducibility/checklist|Reproducibility checklist]]
- [[papers/reproducibility/implementation-readiness|Implementation readiness]]
- [[papers/reproducibility/reproduction-plan|Reproduction plan]]
- [[papers/reproducibility/reproduction-result|Reproduction result]]
- [[papers/analysis/paper-comparison-matrix|Paper comparison matrix]]
- [[concepts/research-methodology/research-question|Research question]]
- [[concepts/research-methodology/literature-synthesis|Literature synthesis]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
