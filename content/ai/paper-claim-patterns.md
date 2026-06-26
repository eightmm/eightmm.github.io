---
title: AI Paper Claim Patterns
tags:
  - ai
  - papers
  - evaluation
---

# AI Paper Claim Patterns

Use this page when a new AI paper looks important but the main contribution is unclear. A paper should be routed by the strongest claim, not by the most recognizable model name.

The minimum AI claim is:

$$
\text{AI claim}
=
(\text{input},\ \text{output},\ \text{model},\ \text{objective},\ \text{evidence})
$$

If the paper also depends on molecules, proteins, structures, or formulas, route it through [[concepts/ai-computational-biology-math-contract|AI Computational Biology Math contract]] after choosing the AI claim pattern.

## Pattern Map

| Pattern | Main Claim | Minimum Evidence | Route |
| --- | --- | --- | --- |
| Architecture claim | a model structure improves quality, cost, or scaling | matched training setup, baseline architecture, ablation, complexity | [Architectures](/ai/architectures) |
| Learning-method claim | a supervision or pretraining signal improves representation or transfer | same architecture where possible, signal definition, transfer protocol | [Learning methods](/ai/learning-methods) |
| Generative-model claim | a distribution model or sampler improves samples | objective, sampler, validity/diversity/utility metrics | [Generative models](/ai/generative-models) |
| Evaluation claim | a benchmark, metric, split, or protocol changes what can be trusted | dataset card, split rule, metric definition, leakage check | [Evaluation](/ai/evaluation) |
| Scaling claim | more data, parameters, compute, memory, or inference budget changes performance | scaling variable, controlled baseline, cost boundary | [Scaling claim contract](/concepts/systems/scaling-claim-contract) |
| Systems claim | implementation, serving, reproducibility, or artifacts make a method practical | runtime evidence, artifact status, reproducibility contract | [Systems](/concepts/systems), [Infra](/infra) |
| Agent claim | tool use, planning, memory, or verification improves task completion | task suite, tool boundary, verifier, failure definition | [Agents](/agents) |

## Architecture Claim

Architecture papers change the function family:

$$
f_\theta \in \mathcal{F}_{\mathrm{new}}
\quad\text{vs}\quad
f_\phi \in \mathcal{F}_{\mathrm{base}}
$$

The claim is not just "new block works." It should identify what bias changed.

| Required | Why |
| --- | --- |
| Input structure | sequence, graph, set, image, coordinate, and multimodal inputs need different biases |
| Complexity | quality claims often trade off with memory, latency, or sequence length |
| Ablation | the new block, connection, normalization, or routing decision must be isolated |
| Matched objective | a different loss can masquerade as an architecture improvement |
| Baseline family | compare against the right CNN, RNN, Transformer, GNN, SSM, MoE, or geometric model |

Useful links:

- [[concepts/architectures/architecture-selection|Architecture selection]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]

## Learning-Method Claim

Learning-method papers change the training signal:

$$
\mathcal{J}(\theta)
=
\mathbb{E}_{u\sim q(u)}
\left[
\ell_\theta^{\mathrm{signal}}(u)
\right]
$$

The paper should say whether the signal is label, mask, contrast, denoising target, preference, reward, pseudo-label, or synthetic target.

| Required | Why |
| --- | --- |
| Signal source | labels, masks, augmentations, preferences, and rewards carry different noise |
| Representation unit | token, sequence, graph, image patch, residue, molecule, or trajectory |
| Adaptation protocol | frozen probe, fine-tuning, retrieval, and full retraining test different claims |
| Transfer split | transfer evidence needs target-domain validation, not only pretraining loss |
| Objective-metric alignment | pretraining objective may not optimize downstream utility |

Useful links:

- [[ai/learning-methods|Learning methods]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/machine-learning/objective-metric-alignment|Objective-metric alignment]]

## Generative-Model Claim

Generative papers change a distribution, path, or sampler:

$$
x \sim p_\theta(x\mid c),
\qquad
\text{sample quality}
\neq
\text{training loss}
$$

The important distinction is learned quantity: likelihood, latent variable, score, noise, velocity, discriminator signal, or consistency map.

| Required | Why |
| --- | --- |
| Learned quantity | score, noise, velocity, likelihood, and reward define different methods |
| Sampling procedure | solver, steps, guidance, filtering, and rejection can own performance |
| Validity rule | syntactic validity, physical validity, and task utility differ |
| Diversity/novelty | high average score can hide mode collapse or near-duplicates |
| Utility metric | final task success may come from a downstream evaluator or filter |

Useful links:

- [[ai/generative-models|Generative models]]
- [[math/formula-patterns|Formula pattern catalog]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]

## Evaluation Claim

Evaluation papers change what evidence means:

$$
\hat{M}
=
\frac{1}{m}
\sum_{j=1}^{m}
M(\hat{y}_j,y_j)
$$

This only supports a claim after the split, model-selection rule, denominator, and uncertainty are known.

| Required | Why |
| --- | --- |
| Example unit | examples, prompts, molecules, proteins, targets, and tasks aggregate differently |
| Split unit | random split, scaffold split, family split, temporal split, and source split test different generalization |
| Selection rule | checkpoint, threshold, prompt, sampler, and filter selection can overfit validation |
| Baseline | a weak or mismatched baseline inflates the contribution |
| Uncertainty | seed variance, confidence interval, and paired comparison determine reliability |

Useful links:

- [[ai/evaluation|Evaluation]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[math/evaluation-math|Evaluation math]]

## Scaling Claim

Scaling papers make performance a function of resource variables:

$$
Q
=
F(D, N, C, B, L)
$$

- $Q$: quality, loss, accuracy, utility, or reward.
- $D$: data size or data quality.
- $N$: parameter count or active parameters.
- $C$: training compute.
- $B$: inference budget, samples, search width, or tool calls.
- $L$: latency, context length, sequence length, or memory budget.

Checks:

- Which variable changed and which variables were controlled?
- Is quality normalized against compute, memory, latency, or cost?
- Does the reported scaling transfer to the target domain or only the benchmark?
- Are inference-time budgets included when comparing methods?

## Systems and Artifact Claim

Systems claims are about making a method usable, reproducible, or cheaper:

$$
\text{method value}
=
\text{quality}
\times
\text{reliability}
\times
\text{deployability}
$$

Minimum evidence:

| Required | Why |
| --- | --- |
| Artifact availability | code, data, splits, configs, weights, and environment must be public or marked missing |
| Runtime boundary | throughput, latency, memory, batch size, hardware class, and precision matter |
| Reproducibility | setup steps and deterministic enough evaluation are part of the claim |
| Failure mode | practical methods need explicit error behavior, not only average speed |

## Agent Claim

Agent papers should not be treated as ordinary model papers if the contribution depends on tools, memory, planning, or verification:

$$
\text{agent}
=
\text{model}
+ \text{state}
+ \text{tools}
+ \text{policy}
+ \text{verifier}
$$

Minimum evidence:

| Required | Why |
| --- | --- |
| Task suite | agents can overfit narrow demonstrations |
| Tool contract | tool permissions, side effects, and outputs define the action space |
| Memory boundary | context, scratchpad, retrieved memory, and persisted memory differ |
| Verification loop | success must be checked by evidence, tests, or human review |
| Failure taxonomy | tool misuse, hallucination, prompt injection, and partial completion need separate handling |

## Final Routing Checklist

- What is the strongest claim: architecture, learning signal, generation, evaluation, scaling, systems, or agent workflow?
- Is the input/output object clear before naming the model?
- Is the objective separate from the final metric?
- Is the evaluation protocol strong enough for the headline claim?
- Does the paper require a domain-specific contract such as Computational Biology?
- Does the paper require a formula-pattern rewrite before summary?
- Are missing metadata, artifacts, or metrics marked `to verify`?

## Related

- [[ai/paper-intake|AI paper intake]]
- [[papers/workflows/claim-routing|Claim routing]]
- [[math/formula-patterns|Formula pattern catalog]]
- [[concepts/coverage-matrix|Coverage matrix]]
- [[concepts/ai-computational-biology-math-contract|AI Computational Biology Math contract]]
