---
title: Benchmark Intake
tags:
  - data
  - benchmark
  - evaluation
  - papers
---

# Benchmark Intake

Benchmark intake is the first pass for deciding what a reported result can actually support. It turns a paper's score into a checked contract over data, task, split, metric, allowed information, and reporting rules.

$$
\text{benchmark claim}
=
(\mathcal{D}, \mathcal{T}, \mathcal{S}, \mathcal{M}, \mathcal{A}, \mathcal{R})
\Rightarrow
\text{supported claim}
$$

where $\mathcal{D}$ is data, $\mathcal{T}$ is the task, $\mathcal{S}$ is the split, $\mathcal{M}$ is the metric set, $\mathcal{A}$ is allowed data/resources/information, and $\mathcal{R}$ is the reporting rule.

## Intake Fields

| Field | Question | Route |
| --- | --- | --- |
| Data | What dataset version, source, filtering, and preprocessing define the examples? | [Dataset card](/concepts/data/dataset-card) |
| Task | What input-output mapping is evaluated? | [Tasks](/concepts/tasks), [Task specification](/concepts/tasks/task-specification) |
| Example unit | What counts as one prediction or one generated sample? | [Example unit](/concepts/data/example-unit) |
| Split unit | What unit is prevented from crossing train, validation, and test? | [Split unit](/concepts/data/split-unit), [Dataset split contract](/concepts/data/dataset-split-contract) |
| Metric | What number decides success, and what diagnostics are secondary? | [Metric selection](/concepts/evaluation/metric-selection) |
| Claim contract | What narrow claim does the score actually support? | [Benchmark claim contract](/concepts/evaluation/benchmark-claim-contract) |
| Baseline | What simple or established method makes the score meaningful? | [Baseline](/concepts/evaluation/baseline) |
| Allowed information | What training data, templates, retrieval corpus, prompts, or preprocessing are allowed? | [Test-set contamination](/concepts/evaluation/test-set-contamination) |
| Reporting | Are uncertainty, paired comparisons, invalid outputs, and subgroup failures reported? | [Evaluation protocol](/concepts/evaluation/evaluation-protocol) |

## Claim Mapping

| Reported Result | Supported Only If |
| --- | --- |
| Higher aggregate score | metric, split, model-selection rule, and uncertainty are comparable |
| Better OOD generalization | held-out units match the claimed deployment shift |
| Better molecular screening | scaffold/source split, early enrichment, negative provenance, and baseline are stated |
| Better protein modeling | sequence/family split, structure source, residue mapping, and template risks are stated |
| Better generative model | validity, diversity, novelty, utility, filtering, and sampling budget are stated |
| Better agent benchmark | task suite, tool access, verifier, human boundary, and failure accounting are stated |

## Evidence Rule

A benchmark score is a finite-sample estimate:

$$
\hat{S}
=
\operatorname{Agg}_{(x_i,y_i)\in \mathcal{D}_{\mathrm{test}}}
m(f_{\theta^\*}(x_i), y_i)
$$

Here $m$ is the per-example metric, $\operatorname{Agg}$ is the aggregation rule, and $\theta^\*$ is the fixed model selected before final test evaluation.

If $\theta^\*$, preprocessing, threshold, prompt, or filtering choices are selected using the test set, the result is no longer an untouched final estimate.

## Failure Checks

- Is the benchmark testing interpolation rather than the claimed shift?
- Are invalid predictions counted in the denominator?
- Are repeated test submissions or leaderboard feedback part of the selection loop?
- Could pretraining, retrieval, templates, or prompt examples contain test items?
- Is the baseline too weak to expose a dataset shortcut?
- Does one aggregate hide per-target, per-scaffold, per-source, or per-domain failures?
- Are uncertainty intervals or paired comparisons needed to interpret the reported gain?

## Bio-Specific Checks

- Molecule benchmarks: standardization, scaffold grouping, decoy provenance, and activity cliffs.
- Protein benchmarks: sequence identity, family split, structure source, residue indexing, and homolog leakage.
- Protein-ligand benchmarks: ligand scaffold, protein family, complex pair, pocket definition, and template leakage.
- Assay benchmarks: endpoint, unit, threshold, censoring, source, and assay harmonization.

## Related

- [[concepts/data/benchmark|Benchmark]]
- [[concepts/data/dataset-card|Dataset card]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[papers/analysis/benchmark-card|Benchmark card]]
- [[ai/paper-intake|AI paper intake]]
- [[bio/paper-intake|Bio paper intake]]
- [[math/formula-intake|Formula intake]]
