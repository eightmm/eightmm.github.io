---
title: Objective-Metric Alignment
tags:
  - machine-learning
  - evaluation
  - objectives
---

# Objective-Metric Alignment

Objective-metric alignment asks whether the training objective optimizes the same quantity that the paper or post uses as evidence. Many AI and molecular modeling claims fail because the loss, selection metric, final metric, and deployment utility are different objects.

The useful decomposition is:

$$
\text{training loss}
\rightarrow
\text{selection metric}
\rightarrow
\text{test metric}
\rightarrow
\text{claimed utility}
$$

Each arrow needs an argument.

## Canonical Form

Training usually minimizes an objective:

$$
\hat{\theta}
=
\arg\min_\theta
\mathbb{E}_{u\sim q_{\mathrm{train}}}
\left[
\ell_\theta(u)
\right]
$$

Evaluation reports a metric:

$$
\hat{M}
=
\frac{1}{n}
\sum_{i=1}^{n}
m(f_{\hat{\theta}}, z_i),
\qquad
z_i \sim \hat{p}_{\mathrm{eval}}
$$

- $u$: unit sampled during training, such as example, token, pair, pose, graph, or trajectory.
- $q_{\mathrm{train}}$: training sampling distribution.
- $\ell_\theta$: optimized loss or surrogate objective.
- $z_i$: evaluation unit.
- $m$: metric contribution.
- $\hat{p}_{\mathrm{eval}}$: evaluation distribution.

The objective and metric align only if $\ell_\theta$, $q_{\mathrm{train}}$, $m$, and $\hat{p}_{\mathrm{eval}}$ support the same decision.

## Common Patterns

| Task | Training Objective | Reported Metric | Alignment Risk |
| --- | --- | --- | --- |
| Classification | cross-entropy | accuracy, F1, AUROC, calibration | probability quality and thresholded decisions differ |
| Regression | MSE, MAE, Gaussian NLL | RMSE, MAE, rank correlation | optimizing scale error may not optimize ranking |
| Ranking | pairwise/listwise loss | NDCG, MAP, enrichment | sampled negatives may not match candidate corpus |
| Retrieval | contrastive loss | recall@k, MRR, enrichment | batch negatives may not match deployment retrieval |
| Generation | likelihood, denoising, score, velocity | validity, diversity, novelty, utility | likelihood or denoising loss may not imply useful samples |
| Docking | pose loss, score loss, diffusion/flow loss | RMSD, pose plausibility, enrichment | valid geometry, ranking, and affinity are separate claims |
| Representation learning | contrastive, masked, predictive loss | linear probe, kNN, fine-tuning | evaluator capacity and adaptation budget change the claim |

## Selection Boundary

Model selection is part of the protocol:

$$
\theta^\*
=
\operatorname*{select}_{\theta \in \Theta}
M_{\mathrm{val}}(f_\theta)
$$

The final test metric is only meaningful if $M_{\mathrm{test}}$ is computed after this selection rule is fixed.

Record:

| Field | Question |
| --- | --- |
| Candidate set | Which checkpoints, seeds, hyperparameters, thresholds, prompts, or filters were considered? |
| Selection metric | Which validation metric chose the final model? |
| Final metric | Which test metric supports the claim? |
| Adaptation budget | Was the representation frozen, linearly probed, fine-tuned, reranked, or filtered? |
| Reporting rule | Are failed runs, invalid samples, or missing labels included? |

## Molecular Modeling Examples

| Claim | Alignment Check |
| --- | --- |
| Better affinity prediction | Does the loss match assay label semantics and the metric used for affinity? |
| Better virtual screening | Does the training objective optimize early enrichment or only pairwise classification? |
| Better pose generation | Is pose RMSD separated from geometry validity and interaction plausibility? |
| Better molecular generation | Are validity, novelty, diversity, and property constraints evaluated after the same filtering rule? |
| Better protein representation | Is the downstream evaluator capacity fixed across methods? |

## Red Flags

- The paper trains with cross-entropy but claims calibrated probabilities without calibration metrics.
- The model is selected by validation AUROC but reported by test PR-AUC under heavy imbalance.
- A generative model reports filtered samples without counting invalid or duplicate samples.
- A docking method uses pose RMSD to imply affinity prediction.
- A representation method compares linear probing against full fine-tuning.
- A metric is averaged across targets, scaffolds, or proteins without showing per-group behavior.

## Related

- [[math/formula-intake|Formula intake]]
- [[concepts/evaluation/claim-evidence-boundary|Claim-evidence boundary]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[bio/data-evaluation|Molecular modeling data and evaluation]]
