---
title: Benchmark Card
unlisted: true
aliases:
  - papers/benchmark-card
tags:
  - papers
  - benchmark
  - evaluation
---

# Benchmark Card

A benchmark card records what a benchmark actually tests before using it to support a paper claim. It is the paper-reading counterpart of a dataset card.

A benchmark can be summarized as:

$$
B = (\mathcal{D}, \mathcal{T}, S, M, P)
$$

$\mathcal{D}$ is data, $\mathcal{T}$ is task, $S$ is split policy, $M$ is metric, and $P$ is preprocessing or protocol.

Use [[concepts/data/benchmark-intake|Benchmark intake]] before writing the card. The intake pass decides whether the reported score is testing the paper's stated claim or only a narrower interpolation setting.

## Fields

- Task: what input-output mapping is evaluated.
- Data source: public dataset or benchmark origin.
- Unit of prediction: molecule, protein, complex, sequence, image, document, or run.
- Split: random, scaffold, protein-family, temporal, source, or held-out target.
- Metric: what number decides success.
- Baseline: what comparison makes the metric meaningful.
- Leakage risks: entity overlap, template leakage, duplicate records, or label leakage.
- Contamination risks: pretraining, retrieval, prompt, feature, model-selection, or leaderboard feedback overlap.
- Saturation status: whether score differences still exceed metric noise.
- Scope: what deployment or scientific claim the benchmark can and cannot support.

## Claim Fit

Benchmark cards should explicitly map score to claim:

| Paper claim | Benchmark must show |
| --- | --- |
| better generalization | split blocks the claimed shortcut |
| better ranking | candidate set and ranking metric match use case |
| better calibration | probability metric and calibration check |
| better generation | validity, novelty, diversity, and task utility |
| better efficiency | latency, throughput, memory, or cost under comparable hardware |
| better scientific fidelity | domain-specific validity checks, not only generic ML metrics |

If the benchmark does not test the claim, route the result to a narrower note instead of repeating the broad claim.

## Benchmark Identity

The same benchmark name can hide protocol differences:

$$
B_{\mathrm{id}}
=
H(\mathcal{D}, S, M, P, \text{version})
$$

Record enough detail to distinguish dataset version, split files, preprocessing, metric implementation, and excluded examples.

## Checks

- Does the split match the generalization claim?
- Does the metric match the task objective?
- Are preprocessing and filtering choices stated?
- Is the final test set protected from contamination?
- Is the benchmark saturated or still discriminative?
- Can a simple baseline expose shortcut learning?
- Are benchmark version and split files identifiable?
- Does the benchmark measure the decision the system will actually make?

## Related

- [[concepts/data/benchmark|Benchmark]]
- [[concepts/data/benchmark-intake|Benchmark intake]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[concepts/evaluation/evaluation-set-design|Evaluation set design]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/test-set-contamination|Test-set contamination]]
- [[concepts/evaluation/benchmark-saturation|Benchmark saturation]]
- [[concepts/sbdd/template-leakage|Template leakage]]
