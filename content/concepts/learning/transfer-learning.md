---
title: Transfer Learning
tags:
  - transfer-learning
  - representation-learning
  - machine-learning
---

# Transfer Learning

Transfer learning reuses knowledge learned on a source task or domain to improve learning on a related target task, typically by reusing pretrained representations.

One way to view it is source pretraining followed by target adaptation:

$$
\theta_s = \arg\min_\theta \mathcal{L}_{\mathrm{source}}(\theta),
\qquad
\theta_t = \arg\min_\theta \mathcal{L}_{\mathrm{target}}(\theta;\theta_s)
$$

The benefit depends on how much useful structure transfers from source to target.

## Source and Target Distributions

Transfer is useful when source and target share structure:

$$
p_{\mathrm{source}}(x,y)
\not=
p_{\mathrm{target}}(x,y)
$$

but the learned representation still captures target-relevant factors:

$$
z
=
\phi_{\theta_s}(x)
$$

If the source task learns features unrelated to the target, transfer can be neutral or harmful.

## Adaptation Modes

Common transfer modes include:

- Frozen feature extraction.
- Linear probing.
- Head-only fine-tuning.
- Full fine-tuning.
- Parameter-efficient fine-tuning.
- Domain adaptation with unlabeled target data.
- Multitask adaptation across related target tasks.

These modes test different claims. A frozen probe tests representation quality. Full fine-tuning tests whether the initialization and architecture adapt well under a given budget.

## Negative Transfer

Negative transfer occurs when source learning hurts target performance:

$$
\hat{R}_{\mathrm{target}}(f_{\mathrm{transfer}})
>
\hat{R}_{\mathrm{target}}(f_{\mathrm{target\ only}})
$$

This can happen when source shortcuts conflict with target labels, when pretraining data overlaps evaluation in misleading ways, or when target data is too different from source data.

## Fair Comparison

Transfer claims should compare against:

$$
\text{target-only baseline}
\quad\text{and}\quad
\text{frozen/probe baseline}
$$

If the transferred model uses more compute, more data, or more hyperparameter trials, the comparison should state that budget difference.

## Why It Matters

- Reduces the labeled data needed on the target task.
- Lets large-scale pretraining amortize across many downstream uses.
- Common in language, vision, and molecular/protein modeling.
- Helps separate representation reuse from task-specific memorization.

## Checks

- How close are the source and target distributions?
- Does the pretrained feature space cover the target's relevant variation?
- Are gains real, or an artifact of overlapping train/test data?
- Is transfer measured with a frozen probe, full fine-tuning, retrieval, or a task-specific evaluator?
- Is there a target-only baseline with the same split and metric?
- Could source pretraining contain target test examples, homologs, duplicates, or templates?
- Does the transfer result hold across data sizes, or only in one label-budget regime?
- Is the adaptation budget comparable across methods?

## Related

- [[concepts/learning/pretraining|Pretraining]]
- [[concepts/learning/domain-adaptation|Domain adaptation]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/linear-probing|Linear probing]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/supervised-learning|Supervised learning]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/baseline|Baseline]]
