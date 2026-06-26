---
title: AI Paper Intake
tags:
  - ai
  - papers
---

# AI Paper Intake

AI 논문을 읽을 때는 모델 이름보다 claim의 위치를 먼저 고정합니다. 같은 Transformer 논문이라도 architecture 논문, learning objective 논문, generative model 논문, evaluation 논문은 서로 다른 축으로 읽어야 합니다.

$$
\text{paper claim}
=
(\text{input}, \text{output}, \text{model}, \text{objective}, \text{evidence})
$$

## Intake Fields

| Field | Question | Route |
| --- | --- | --- |
| Input and output | What object enters the model, and what object is predicted? | [Modalities](/concepts/modalities), [Tasks](/concepts/tasks) |
| Prediction type | Is it classification, regression, ranking, retrieval, generation, or action selection? | [Machine Learning](/ai/machine-learning) |
| Architecture | What inductive bias, parameter sharing, and complexity does the model use? | [Architectures](/ai/architectures) |
| Learning signal | Is the signal label, mask, contrast, denoising target, preference, reward, or synthetic target? | [Learning Methods](/ai/learning-methods) |
| Distribution model | Does the method model $p_\theta(x)$, $p_\theta(y\mid x)$, a score, or a vector field? | [Generative Models](/ai/generative-models), [Math](/math) |
| Evidence | Which metric, split, baseline, ablation, and uncertainty support the claim? | [Evaluation](/ai/evaluation) |
| Benchmark claim | What data, task, split, metric, allowed information, and reporting rule define the score? | [Benchmark intake](/concepts/data/benchmark-intake) |
| System boundary | Does the contribution depend on data scale, compute, serving, tools, or reproducibility? | [Systems](/concepts/systems), [Infra](/infra), [Agents](/agents) |

## Claim Shapes

| Claim Shape | Minimum Evidence |
| --- | --- |
| Better architecture | matched objective, matched training budget, baseline architecture, ablation, complexity or throughput note |
| Better learning method | same architecture where possible, pretraining data description, downstream transfer protocol, frozen or fine-tuned evaluation |
| Better generative model | likelihood or surrogate objective, sampling procedure, validity/diversity/novelty/utility metrics, failure examples |
| Better benchmark result | exact split, model-selection rule, primary metric, uncertainty or seed variation, leakage check |
| Better agent workflow | task suite, tool boundary, verifier, success/failure definition, human review boundary |

## Formula Checks

When a paper introduces an objective, rewrite it into these parts:

$$
\hat{\theta}
=
\arg\min_\theta
\mathbb{E}_{(x,y)\sim p_{\mathrm{train}}}
\left[
\mathcal{L}_\theta(x,y)
\right]
$$

- Distribution: what does the expectation sample from?
- Target: what is observed, masked, generated, preferred, or rewarded?
- Parameter: which module is optimized?
- Decision: how does the trained score or probability become an output?
- Evaluation: is the final metric the same object as the training loss?

## Update Targets

After reading, update the smallest durable note that captures the reusable idea.

- Architecture idea: [[ai/architectures|Architectures]] or [[concepts/architectures/index|Architecture concepts]]
- Learning signal: [[ai/learning-methods|Learning Methods]] or [[concepts/learning/index|Learning concepts]]
- Generative objective: [[ai/generative-models|Generative Models]] or [[concepts/generative-models/index|Generative model concepts]]
- Evaluation issue: [[ai/evaluation|Evaluation]] or [[concepts/evaluation/index|Evaluation concepts]]
- Benchmark issue: [[concepts/data/benchmark-intake|Benchmark intake]] or [[papers/analysis/benchmark-card|Benchmark card]]
- Paper-specific claim: [[papers/analysis/claim-extraction|Claim extraction]] and [[papers/analysis/evidence-table|Evidence table]]
- Multi-axis paper: [[papers/workflows/claim-routing|Claim routing]]

## Related

- [[ai/index|AI]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[papers/workflows/claim-routing|Claim routing]]
- [[math/formula-intake|Formula intake]]
- [[molecular-modeling/paper-intake|Molecular modeling paper intake]]
