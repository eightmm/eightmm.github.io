---
title: Benchmark Claim Contract
tags:
  - evaluation
  - benchmark
  - papers
---

# Benchmark Claim Contract

Benchmark claim contract maps a reported score to the narrow claim it can support. Use it when a paper says a method is better, more general, more useful, or more robust.

$$
\text{claim support}
=
g(\mathcal{D}, \mathcal{S}, \mathcal{T}, \mathcal{M}, b, u)
$$

where $\mathcal{D}$ is the dataset, $\mathcal{S}$ is the split, $\mathcal{T}$ is the task, $\mathcal{M}$ is the metric set, $b$ is the baseline, and $u$ is uncertainty or variance.

## Contract Fields

| Field | Must State | Why It Matters |
| --- | --- | --- |
| Dataset | name, version, source, filtering, preprocessing | defines the population being measured |
| Example unit | one molecule, protein, complex, image, prompt, trajectory, or generated sample | prevents mismatched denominators |
| Split unit | random row, scaffold, protein family, time, source, complex pair, or task instance | defines the generalization boundary |
| Task | input, output, validity rule, allowed context | separates classification, ranking, generation, and coordinate claims |
| Metric | primary metric, diagnostics, aggregation, invalid-output policy | states what success means |
| Selection rule | validation metric, threshold choice, checkpoint choice, prompt choice, filtering rule | separates model selection from final evidence |
| Baseline | simple baseline, previous method, oracle upper bound, or ablation | makes the score interpretable |
| Uncertainty | seed variance, bootstrap interval, paired comparison, subgroup variance | prevents overreading small differences |
| Allowed information | pretraining data, templates, retrieval corpus, tools, prompts, featurizers | detects contamination and hidden assistance |
| Compute budget | data scale, model size, training steps, hardware, precision, inference budget | separates quality claims from resource changes |

## Score to Claim Map

| Reported Score | Conservative Claim |
| --- | --- |
| Higher IID test score | better on this dataset distribution under this protocol |
| Higher scaffold split score | better chemical-series generalization if scaffold grouping is valid |
| Higher protein-family split score | better target-family generalization if homolog leakage is controlled |
| Higher pose success | better pose placement under the same receptor, ligand, and symmetry policy |
| Higher docking enrichment | better early ranking under this active/decoy construction |
| Higher generation validity | fewer invalid samples under this sampler and filtering policy |
| Higher benchmark average | better aggregate score, not necessarily better on every subgroup |
| Better leaderboard rank | better under leaderboard rules, not necessarily better deployment utility |

## Claim Formula

A paper's benchmark table usually reports an estimate:

$$
\hat{\Delta}
=
\operatorname{Agg}_{i\in \mathcal{D}_{test}}
\left[
m(f_{\theta^\*}(x_i), y_i)
-
m(f_{\phi^\*}(x_i), y_i)
\right]
$$

where $f_{\theta^\*}$ is the proposed model, $f_{\phi^\*}$ is the baseline, $m$ is the per-example metric, and $\operatorname{Agg}$ is the aggregation rule.

The claim is weak when $\theta^\*$, $\phi^\*$, thresholds, prompts, filtering, or preprocessing are chosen after looking at the final test set.

## Molecular Modeling Additions

| Area | Extra Check |
| --- | --- |
| Molecule property | standardization, scaffold split, assay/source split, activity cliffs |
| Protein model | sequence identity, family split, residue indexing, structure source |
| Protein-ligand complex | ligand scaffold, protein family, pair split, pocket definition, template leakage |
| Docking and pose | atom mapping, symmetry correction, receptor preparation, ligand protonation, pose validity |
| Virtual screening | active/decoy provenance, early enrichment, duplicate handling, target leakage |
| Conformer model | conformer source, ensemble size, energy ranking, downstream representation shift |

## Stop Before Claiming

- The dataset version or filtering policy is missing.
- The split does not match the claimed deployment shift.
- The baseline uses different data, examples, or allowed information.
- The proposed method uses a larger compute, data, model-selection, or inference budget without stating it.
- The primary metric was chosen after seeing many alternatives.
- Invalid generations or failed predictions are removed without a denominator.
- Only aggregate performance is shown while subgroup failures are plausible.
- The result is within seed variance or confidence intervals.
- The table reports a best seed or best checkpoint without the selection rule.
- The paper's evidence supports a benchmark-specific claim, but the draft says the method is generally superior.

## Related

- [[concepts/data/benchmark-intake|Benchmark intake]]
- [[concepts/data/dataset-card|Dataset card]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/claim-evidence-boundary|Claim-evidence boundary]]
- [[concepts/evaluation/seed-variance|Seed variance]]
- [[papers/analysis/result-table-reading|Result table reading]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
- [[molecular-modeling/data-evaluation|Molecular modeling data and evaluation]]
