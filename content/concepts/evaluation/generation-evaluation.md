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

## Evaluation Axes

| Axis | Question | Typical Evidence |
| --- | --- | --- |
| validity | does the output obey syntax, chemistry, geometry, physics, or executable rules? | validity rate and failure taxonomy |
| fidelity | does the output satisfy the prompt, condition, property, pocket, scaffold, or constraint? | condition-specific metric and verifier |
| diversity | does the model cover varied outputs under the same condition? | uniqueness, pairwise distance, cluster coverage |
| novelty | is the output not memorized or near-duplicate? | nearest-neighbor distance under the right equivalence relation |
| utility | does the output help the downstream task? | independent downstream assay, simulation, verifier, or benchmark |
| cost | how expensive is one useful output? | NFE, candidates generated, candidates kept, wall-clock, hardware |

## Sample Accounting

Report metrics with the denominator visible:

$$
\mathrm{useful\ rate}
=
\frac{
\#\{\text{valid, novel, condition-satisfying, useful samples}\}
}{
\#\{\text{attempted samples}\}
}
$$

If a method reports only post-filtered samples:

$$
\mathrm{metric}_{\mathrm{kept}}
\neq
\mathrm{metric}_{\mathrm{attempted}}
$$

The difference matters for sampling cost and for whether the model or the filter produced the quality.

## Molecular and Structure Generation

| Object | Additional Checks |
| --- | --- |
| molecule | chemical validity, duplicate handling, scaffold novelty, property predictor bias |
| conformer | RMSD/diversity, energy or clash checks, stereochemistry, atom mapping |
| protein sequence | sequence validity, family novelty, function proxy, developability constraints |
| protein structure | geometry, clashes, secondary structure, fold novelty, confidence or relaxation protocol |
| protein-ligand complex | pose validity, pocket context, interaction quality, ligand/protein split, template leakage |

## Checks

- Is the task open-ended or reference-based?
- Are outputs required to be syntactically valid, physically valid, executable, or factual?
- Does the metric reward generic safe outputs too much?
- Is diversity measured together with quality?
- Is there a verifier, downstream task, or human review protocol?
- Are invalid, duplicate, or rejected samples included in the denominator?
- Is novelty measured against training data, public databases, or only the test set?
- Is the evaluator independent from guidance, filtering, and model selection?

## Related

- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/generative-models/index|Generative models]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/guidance|Guidance]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/llm/hallucination-grounding|Hallucination and grounding]]
- [[agents/verification/verification-loop|Verification loop]]
- [[concepts/sbdd/pose-quality|Pose quality]]
