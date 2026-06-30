---
title: Hypothesis Testing
tags:
  - math
  - statistics
  - evaluation
---

# Hypothesis Testing

Hypothesis testing is a statistical procedure for comparing observed evidence against a null hypothesis. In model evaluation, it is often used to ask whether an observed metric difference is plausibly larger than evaluation noise.

A test starts with:

$$
H_0: \theta \in \Theta_0
\qquad
\text{vs.}
\qquad
H_1: \theta \in \Theta_1
$$

where $H_0$ is the null hypothesis and $H_1$ is the alternative.

## Test Statistic

A test statistic $T(X)$ summarizes the data into a value whose distribution under $H_0$ is known or approximated:

$$
T_{\mathrm{obs}} = T(x_1,\ldots,x_n)
$$

The $p$-value is:

$$
p
=
P_{H_0}
\left(
T(X) \ge T_{\mathrm{obs}}
\right)
$$

for a one-sided test. Two-sided tests use an extremeness rule appropriate for both directions.

The $p$-value is not:

$$
P(H_0 \mid \text{data})
$$

It is a probability of seeing a statistic at least as extreme as the observed one under the null model.

## Errors

Type I error rejects a true null:

$$
\alpha = P(\text{reject }H_0 \mid H_0 \text{ true})
$$

Type II error fails to reject a false null:

$$
\beta = P(\text{fail to reject }H_0 \mid H_1 \text{ true})
$$

Power is:

$$
1-\beta
$$

## Paired Model Comparison

Model evaluation is often paired: two models are evaluated on the same examples. Let $m_A(x_i)$ and $m_B(x_i)$ be per-example scores, losses, or correctness indicators, and define:

$$
d_i = m_A(x_i)-m_B(x_i)
$$

Then the question is whether the mean paired difference is meaningfully different from zero:

$$
H_0:\mathbb{E}[d]=0
$$

A paired test or bootstrap over examples is usually more appropriate than treating two aggregate metrics as independent numbers.

## Multiple Comparisons

If many models, prompts, seeds, checkpoints, thresholds, or benchmarks are tried, the chance of at least one false positive increases:

$$
P(\text{at least one false positive})
=
1-(1-\alpha)^m
$$

where $m$ is the number of independent tests. In paper reading, this is a warning to check whether the selection rule was fixed before the final evaluation.

## Effect Size

Statistical significance and practical significance are different. A useful evaluation note should report both uncertainty and magnitude:

$$
\widehat{\Delta}
=
\widehat{M}_A-\widehat{M}_B
$$

with a confidence interval or bootstrap distribution when possible. A tiny but significant improvement may not matter if the benchmark is saturated, the cost is higher, or the error mode is unchanged.

## Practical Meaning

Rejecting $H_0$ does not prove that a model is useful. It only says the observed statistic is unlikely under the chosen null model. Practical value still depends on effect size, evaluation protocol, uncertainty, and deployment relevance.

## Evaluation Checklist

| Question | Why |
| --- | --- |
| What is the independent unit? | examples, molecules, proteins, targets, prompts, or users may be correlated |
| Is the comparison paired? | same examples reduce variance and change the correct test |
| Was the test chosen before looking? | post-hoc testing inflates claims |
| How many comparisons were tried? | model selection and threshold search create false positives |
| Is the effect size meaningful? | significance alone does not prove usefulness |
| Does the metric match the claim? | a significant proxy metric may not support the stated behavior |

## Checks

- What is the unit of independence?
- Is the test paired or unpaired?
- Is the test one-sided or two-sided?
- Was the test chosen before seeing the result?
- Are multiple comparisons controlled when many models, prompts, thresholds, or seeds are tried?
- Is the effect size meaningful, not only statistically significant?
- Is the final test separated from model selection, threshold tuning, and prompt selection?

## Related

- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/bootstrap-evaluation|Bootstrap evaluation]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/effect-size|Effect size]]
- [[concepts/evaluation/multiple-comparisons|Multiple comparisons]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/research-methodology/hypothesis|Hypothesis]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
