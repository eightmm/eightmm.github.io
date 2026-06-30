---
title: AI Computational Biology Math Contract
tags:
  - concepts
  - ai
  - molecular-modeling
  - computational-biology
  - math
---

# AI Computational Biology Math Contract

Use this contract when a paper, post, or project combines an AI method, a computational biology object, and a mathematical objective or metric.

The minimum useful record is:

$$
\text{claim}
=
(\text{object},\ \text{representation},\ \text{model},\ \text{objective},\ \text{evidence})
$$

If one part is missing, the note may still be useful, but the missing part should be marked `to verify` rather than inferred.

## Axis Roles

| Axis | Owns | Does Not Own |
| --- | --- | --- |
| Computational Biology | molecule, ligand, protein, pocket, complex, conformer, genome region, assay, split unit | generic architecture definitions |
| AI | architecture, learning method, generative model, training protocol, inference behavior | biological label semantics by itself |
| Math | objects, indices, distributions, objectives, metrics, uncertainty, symmetry | paper-specific biological interpretation by itself |
| Evaluation | split, metric, baseline, selection rule, leakage, uncertainty, allowed information | headline score without protocol |
| Systems | compute, memory, throughput, artifacts, reproducibility, serving boundary | model quality claim without task evidence |

## Minimum Fields

| Field | Required Question | Route |
| --- | --- | --- |
| Object | What biological or chemical unit is modeled? | [Computational Biology](/molecular-modeling), [Entities](/entities) |
| Context | Does the claim depend on target, pocket, assay, receptor state, species, construct, or template? | [Computational Biology Boundary](/molecular-modeling/computational-biology) |
| Representation | How does the raw object become tokens, graph, coordinates, embedding, or features? | [Representation contract](/concepts/modalities/representation-contract) |
| Architecture | What inductive bias, parameter sharing, and complexity are used? | [Architectures](/ai/architectures) |
| Learning signal | What supervision, mask, contrast, denoising target, preference, or reward is used? | [Learning methods](/ai/learning-methods) |
| Objective | What loss, likelihood, score, velocity, estimator, or constraint is optimized? | [Formula intake](/math/formula-intake) |
| Metric | What score is reported, and does it match the objective? | [Objective-metric alignment](/concepts/machine-learning/objective-metric-alignment) |
| Split | What unit is held out: scaffold, protein family, complex pair, assay, source, time, or template group? | [Data and evaluation](/molecular-modeling/data-evaluation) |
| Evidence | Which benchmark, baseline, ablation, uncertainty, and failure mode support the claim? | [Benchmark claim contract](/concepts/evaluation/benchmark-claim-contract) |
| Public boundary | Can the note be published without private data, internal paths, unpublished results, or collaborator details? | [Publishing gate](/inbox/publishing-gate) |

## Formula Pattern

Most mixed AI and computational biology papers can be rewritten as:

$$
\hat{\theta}
=
\arg\min_\theta
\mathbb{E}_{u \sim q(u)}
\left[
w(u)\,
\mathcal{L}_\theta
\left(
r(u), y(u), c(u)
\right)
\right]
$$

- $u$: sampled unit, such as molecule, protein, complex, conformer, residue, atom, assay row, or sequence window.
- $q(u)$: training sampling distribution.
- $w(u)$: mask, class weight, importance weight, time weight, or balancing rule.
- $r(u)$: representation passed to the model.
- $y(u)$: label, target, generated object, score, velocity, noise, reward, or preference.
- $c(u)$: context such as target, pocket, assay, condition, source, or time.
- $\theta$: optimized model parameters.

This rewrite forces the note to separate biological object, computational representation, learning objective, and evidence.

## Routing Decisions

| Main Claim | Primary Home | Required Companions |
| --- | --- | --- |
| New model block improves performance | [AI architectures](/ai/architectures) | object contract, objective, benchmark boundary |
| New pretraining or SSL signal helps transfer | [Learning methods](/ai/learning-methods) | representation unit, transfer protocol, split rule |
| New generative method samples valid objects | [Generative models](/ai/generative-models) | validity metric, sampling procedure, utility metric |
| New docking, conformer, or structure workflow | [Computational Biology](/molecular-modeling) | preprocessing, coordinate contract, leakage check |
| New formula or estimator explains the method | [Math](/math) | symbol table, sampled distribution, evaluation link |
| New benchmark or leaderboard result is central | [Evaluation](/ai/evaluation) | dataset card, split, baseline, uncertainty |
| New implementation makes the method practical | [Infra](/infra) | artifact availability, reproducibility, scaling boundary |

## Cross-Axis Failure Modes

| Failure Mode | Symptom | Fix |
| --- | --- | --- |
| model name hides object | "Transformer for molecules" without molecule state or label context | add object and representation contract |
| biology hides method | "docking model" without architecture, objective, or sampler | route method details through AI and Math |
| formula hides data | clean loss equation without sampled unit or split | add $u$, $q(u)$, preprocessing, and split unit |
| metric hides objective | training loss and reported metric measure different things | add objective-metric alignment |
| benchmark hides leakage | high score with row split or unclear allowed information | add split, leakage, and allowed-information check |
| public post hides uncertainty | Korean synthesis states paper claim as fact | mark claim level and evidence boundary |

## Claim Record

Use this compact record inside paper notes or posts:

```yaml
claim:
  text: to verify
  primary_axis: ai | computational-biology | math | evaluation | systems | agents
  object: to verify
  representation: to verify
  model: to verify
  objective: to verify
  metric: to verify
  split: to verify
  evidence_level: mentioned
  public_boundary: to verify
```

This keeps the note from becoming a loose summary. Every major statement should be traceable to one claim record or to a reusable wiki definition.

## Readiness Checks

- The title makes the primary axis clear.
- The biological object is not hidden behind only the model name.
- The model input is distinguished from the raw object.
- The objective is written with sampled unit, distribution, and optimized parameter.
- The reported metric and selection rule are named.
- The split unit matches the generalization claim.
- Coordinate and symmetry behavior are stated for 3D claims.
- Public artifact status is `released`, `not released`, `to verify`, or `not applicable`.
- Missing metadata stays `to verify`.
- The public note does not expose private infrastructure, paths, people, tasks, or unpublished results.

## Related

- [[ai/paper-intake|AI paper intake]]
- [[molecular-modeling/paper-intake|Computational Biology paper intake]]
- [[math/formula-intake|Formula intake]]
- [[papers/workflows/claim-routing|Claim routing]]
- [[papers/workflows/ai-molecular-math-paper-template|AI computational biology math paper template]]
- [[papers/workflows/ai-molecular-math-readiness-gate|AI computational biology math readiness gate]]
- [[posts/workflows/ai-molecular-math-post-intake|AI computational biology math post intake]]
- [[concepts/coverage-matrix|Coverage matrix]]
