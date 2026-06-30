---
title: Generative Model Papers
tags:
  - papers
  - generative-models
---

# Generative Model Papers

Generative model paper note는 sample, transform, denoise, decode, structured object generation을 다루는 논문을 모읍니다.

이 선반의 기준은 "새 sample을 만들거나 distribution을 모델링하는가"입니다.

$$
x \sim p_\theta(x \mid c)
$$

where $c$ can be text, class label, protein pocket, scaffold, structure constraint, trajectory time, or another conditioning signal.

## Reading Axes

- 어떤 distribution을 모델링하는가?
- generation 방식이 autoregressive, diffusion-based, flow-based, energy-based, latent-variable 중 무엇인가?
- 어떤 objective를 optimize하는가?
- validity, diversity, novelty, task utility를 어떻게 평가하는가?
- model이 text, molecule, protein, structure, image, action 중 무엇을 생성하는가?
- claim이 likelihood, sample quality, controllability, efficiency, downstream utility 중 무엇에 관한 것인가?

## Routing

| Strongest Claim | Put Here? | Cross-Link |
| --- | --- | --- |
| new diffusion, flow, autoregressive, latent, or energy objective | yes | [Generative models](/concepts/generative-models) |
| molecule or protein generator | yes, if generation is the core contribution | [Computational Biology papers](/papers/computational-biology) |
| docking or pose search with generative component | sometimes | [Structure-based modeling papers](/papers/sbdd) |
| architecture block for many tasks | no, unless generation evidence is central | [Architecture papers](/papers/architectures) |
| LLM decoding or instruction generation | usually no | [Architecture papers](/papers/architectures) |

## Evaluation Boundary

Generation papers often report many metrics that answer different questions.

| Metric Family | Asks | Common Trap |
| --- | --- | --- |
| validity | is the output syntactically/physically valid? | invalid denominator removed after filtering |
| diversity | are samples different from each other? | high diversity with low utility |
| novelty | are samples unlike training data? | novelty defined by weak similarity threshold |
| quality | does a generated sample look or score well? | evaluator model shares training bias |
| controllability | does condition $c$ change output as intended? | condition leakage or easy constraints |
| downstream utility | does generated output improve a task? | cherry-picked candidates or missing baseline |
| efficiency | is sampling cheaper or faster? | hardware, batch size, and sampler budget hidden |

## Claim Extraction

For each paper, separate:

$$
\text{objective}
\rightarrow
\text{sampler}
\rightarrow
\text{filter}
\rightarrow
\text{metric}
\rightarrow
\text{claim}
$$

| Field | Question |
| --- | --- |
| Objective | likelihood, denoising, score, velocity, energy, reward, reconstruction? |
| Sampler | how many steps, candidates, retries, guidance, temperature, or constraints? |
| Filter | are invalid or low-quality samples removed before scoring? |
| Condition | what information is available at generation time? |
| Baseline | same budget, same filter, same evaluator, same dataset? |
| Utility | is the generated object useful under an independent check? |

## Concepts

- [[concepts/generative-models/index|Generative models]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/generative-models/protein-design|Protein design]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]

## Curated Notes

- [[papers/architectures/auto-encoding-variational-bayes|Auto-Encoding Variational Bayes]]
- [[papers/architectures/generative-adversarial-nets|Generative Adversarial Nets]]
- [[papers/architectures/real-nvp|Real NVP]]
- [[papers/architectures/ddpm|Denoising Diffusion Probabilistic Models]]
- [[papers/architectures/latent-diffusion-models|Latent Diffusion Models]]
- [[papers/generative-models/molexar|Molexar]]

## Related

- [[ai/generative-models|Generative models]]
- [[papers/computational-biology/index|Computational Biology papers]]
- [[concepts/evaluation/metric|Metric]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/analysis/benchmark-card|Benchmark card]]
- [[papers/analysis/paper-comparison-matrix|Paper comparison matrix]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
