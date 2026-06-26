---
title: Multiple Comparisons
tags:
  - evaluation
  - statistics
  - methodology
---

# Multiple Comparisons

Multiple comparisons occur when many models, prompts, thresholds, seeds, datasets, metrics, or ablations are tried and only the best-looking result is emphasized. The more comparisons are made, the easier it is to find a false positive.

If each comparison has false-positive probability $\alpha$ and $K$ independent comparisons are tried, the chance of at least one false positive is:

$$
P(\text{at least one false positive})
= 1 - (1-\alpha)^K
$$

The independence assumption is often imperfect, but the direction is the same: more unreported attempts make a selected result less trustworthy.

## Where It Appears

- Hyperparameter sweeps where only the best trial is reported.
- Many random seeds with the best seed selected.
- Prompt engineering against a public benchmark.
- Many ablations, metrics, or datasets with selective reporting.
- Threshold tuning on the final test set.
- Leaderboard submissions used as iterative feedback.

## Controls

- Predefine the primary metric and selection rule.
- Separate validation from final test evaluation.
- Report the search space, number of trials, failed trials, and seed policy.
- Use held-out confirmation for the final selected model.
- Adjust interpretation when many comparisons were attempted.
- Treat exploratory ablations as hypothesis generation unless confirmed.

## Bonferroni Bound

A conservative threshold divides the target false-positive rate by the number of tests:

$$
\alpha_{\mathrm{per-test}} = \frac{\alpha_{\mathrm{family}}}{K}
$$

$K$ is the number of planned comparisons. This is simple but can be too conservative when tests are correlated; the important habit is to disclose and separate exploration from confirmation.

## Checks

- How many variants, prompts, thresholds, seeds, or metrics were tried?
- Was the final test set used once, after model selection?
- Are negative and failed runs visible?
- Is the claim exploratory or confirmatory?
- Would the result still look meaningful under a stricter threshold or held-out confirmation?

## Related

- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/effect-size|Effect size]]
- [[concepts/evaluation/test-set-contamination|Test-set contamination]]
- [[concepts/machine-learning/hyperparameter-tuning|Hyperparameter tuning]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/research-methodology/threat-to-validity|Threat to validity]]
