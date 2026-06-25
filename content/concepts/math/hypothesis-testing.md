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

## Practical Meaning

Rejecting $H_0$ does not prove that a model is useful. It only says the observed statistic is unlikely under the chosen null model. Practical value still depends on effect size, evaluation protocol, uncertainty, and deployment relevance.

## Checks

- What is the unit of independence?
- Is the test paired or unpaired?
- Is the test one-sided or two-sided?
- Was the test chosen before seeing the result?
- Are multiple comparisons controlled when many models, prompts, thresholds, or seeds are tried?
- Is the effect size meaningful, not only statistically significant?

## Related

- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/bootstrap-evaluation|Bootstrap evaluation]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/research-methodology/hypothesis|Hypothesis]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
