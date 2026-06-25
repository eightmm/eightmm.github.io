---
title: Generation Evaluation
tags:
  - evaluation
  - generation
  - metrics
---

# Generation Evaluation

Generation evaluation measures whether produced outputs are valid, useful, faithful, diverse, and aligned with the intended task. It is harder than classification or regression because many outputs can be acceptable.

For conditional generation, the model samples:

$$
\hat{y}
\sim
p_\theta(y\mid x)
$$

Evaluation usually combines automatic metrics, validity checks, downstream tests, and human or verifier judgment.

## Likelihood-Based Evaluation

Negative log-likelihood evaluates assigned probability to reference outputs:

$$
\operatorname{NLL}
=
-\frac{1}{n}
\sum_{i=1}^{n}
\log p_\theta(y_i\mid x_i)
$$

Per-token perplexity is:

$$
\operatorname{PPL}
=
\exp
\left(
\frac{1}{T}
\sum_{t=1}^{T}
-\log p_\theta(y_t\mid y_{<t},x)
\right)
$$

Likelihood can miss usefulness when many valid outputs differ from the reference.

## Constraint and Validity Checks

Generated objects often need hard validity checks:

$$
\operatorname{Validity}
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf{1}[\hat{y}_i \in \mathcal{Y}_{\mathrm{valid}}]
$$

Here $\mathcal{Y}_{\mathrm{valid}}$ can mean valid syntax, executable code, chemically valid SMILES, physically plausible pose, or supported answer.

## Diversity

One simple uniqueness estimate is:

$$
\operatorname{Unique}
=
\frac{
|\{\hat{y}_i\}_{i=1}^{n}|
}{n}
$$

Diversity must be interpreted with quality; random invalid outputs can be diverse but useless.

## Checks

- Is the task open-ended or reference-based?
- Are outputs required to be syntactically valid, physically valid, executable, or factual?
- Does the metric reward generic safe outputs too much?
- Is diversity measured together with quality?
- Is there a verifier, downstream task, or human review protocol?

## Related

- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/generative-models/index|Generative models]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/llm/hallucination-grounding|Hallucination and grounding]]
- [[agents/verification/verification-loop|Verification loop]]
- [[concepts/sbdd/pose-quality|Pose quality]]
