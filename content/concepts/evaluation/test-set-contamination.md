---
title: Test-Set Contamination
tags:
  - evaluation
  - leakage
  - benchmark
---

# Test-Set Contamination

Test-set contamination occurs when information about final evaluation examples enters training, pretraining, retrieval, model selection, prompt construction, preprocessing, or human iteration. It is a specific form of [[concepts/evaluation/leakage|Leakage]] focused on the final test boundary.

A strict test boundary asks for:

$$
I(\mathcal{D}_{\mathrm{test}}; A_{\mathrm{train}}) = 0
$$

$I(\cdot;\cdot)$ is mutual information and $A_{\mathrm{train}}$ is the full training procedure: data, preprocessing, prompts, retrieval corpus, hyperparameter search, checkpoint selection, thresholds, and manual decisions. In practice the goal is not literal zero information, but no deployment-unavailable test-specific information.

## Channels

- Exact duplicate examples in train and test.
- Near-duplicates, paraphrases, homologs, scaffold analogs, template structures, or repeated prompts.
- Pretraining corpus includes benchmark questions, answers, labels, or target structures.
- Retrieval index includes final test examples or answer keys.
- Feature engineering uses statistics fit on the full dataset.
- Hyperparameter tuning, threshold selection, checkpoint choice, or failed-run exclusion uses final test feedback.
- Public leaderboard submissions repeatedly shape the model or prompt.

## Boundary Test

Write the final test contract as:

$$
\mathcal{D}_{\mathrm{test}} \cap_{\mathrm{info}} \mathcal{A}_{\mathrm{train}} = \varnothing
$$

$\cap_{\mathrm{info}}$ means information overlap, not only identical rows. For molecules this can include scaffold or near-neighbor overlap; for proteins, homolog or template overlap; for LLM tasks, prompt-answer overlap; for retrieval systems, corpus-answer overlap.

## Audit Checklist

- Deduplicate exact examples before splitting.
- Check near-duplicates using domain-appropriate similarity.
- Check split groups: user, source, scaffold, protein family, document, prompt family, time, or benchmark origin.
- Verify that preprocessing, normalization, imputation, feature selection, and label conversion are fit only on training data.
- Separate validation from final test for model selection.
- Record how many times the final benchmark was inspected or submitted.
- For public benchmarks, state whether benchmark-aware prompt engineering or training is allowed.

## Interpretation

Contamination does not always make a result useless, but it narrows the claim. A contaminated score may still debug an implementation, compare within a known benchmark setting, or measure memorization. It should not be used as evidence for new-data generalization.

## Related

- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/evaluation-set-design|Evaluation set design]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/evaluation/benchmark-saturation|Benchmark saturation]]
- [[concepts/sbdd/template-leakage|Template leakage]]
