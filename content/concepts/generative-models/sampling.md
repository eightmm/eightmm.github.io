---
title: Sampling
tags:
  - generative-models
  - inference
  - sampling
---

# Sampling

Sampling is the procedure that turns a learned generative model into concrete outputs.

For an unconditional model:

$$
x \sim p_\theta(x)
$$

For a conditional model:

$$
x \sim p_\theta(x \mid c)
$$

The sampling algorithm determines quality, diversity, latency, and controllability.

## Sampling Patterns

| Pattern | Base Randomness | Main Control | Common Cost |
| --- | --- | --- | --- |
| autoregressive sampling | token-level categorical draws | temperature, top-k, top-p, beam, constraints | one model call per step or block |
| diffusion sampling | Gaussian noise and denoising path | noise schedule, sampler, guidance, NFE | many denoising evaluations |
| flow sampling | base noise and ODE/invertible transform | solver, step count, path, guidance | ODE or inverse transform cost |
| latent sampling | latent $z\sim p(z)$ | prior, decoder, latent conditioning | decoder plus latent search/filtering |
| rejection or filtering | candidate pool | verifier, score, validity rule, threshold | many candidates per kept sample |

Here NFE means number of function evaluations. It is often the right unit for diffusion, score, and flow samplers:

$$
\mathrm{cost}_{\mathrm{sample}}
\approx
\mathrm{NFE}
\times
\mathrm{cost}(f_\theta)
+
\mathrm{cost}(\mathrm{filter})
$$

## Diversity and Quality

Sampling often trades diversity for fidelity. If the sampler collapses to high-probability modes, outputs may look good but cover only a small part of the data distribution.

## Effective Sample Distribution

Filtering, reranking, guidance, and beam search change the distribution being evaluated:

$$
x \sim p_{\theta,\mathrm{sampler}}(x\mid c)
\neq
p_\theta(x\mid c)
$$

For post-filtering with acceptance rule $A(x)=1$:

$$
p_{\mathrm{kept}}(x\mid c)
\propto
p_{\theta,\mathrm{sampler}}(x\mid c)\,\mathbf{1}[A(x)=1]
$$

Reported validity, novelty, and utility should state whether they are computed before filtering, after filtering, or per kept sample.

## Reporting Fields

| Field | Required Detail |
| --- | --- |
| sampler | decoding rule, ODE/SDE solver, denoising schedule, or latent prior |
| stochasticity | seed policy, temperature, noise scale, top-k/top-p, restart count |
| guidance | guidance type, strength, schedule, and guide source |
| filtering | validity checks, score thresholds, rejection rate, reranking rule |
| budget | candidates generated, candidates kept, NFE, wall-clock, hardware, memory |
| distribution | whether metrics are pre-filter, post-filter, best-of-n, or oracle-selected |

## Checks

- What distribution is sampled first: token, noise, latent, graph, or coordinates?
- How many sampling steps or decoding calls are required?
- Are temperature, guidance strength, beam size, or rejection filters documented?
- Are invalid samples counted or silently removed?
- Are diversity, novelty, and task utility evaluated separately?
- Are best-of-n and reranking reported as part of the method rather than hidden evaluation?
- Is the comparison matched by quality at equal NFE/candidate budget, or only by headline metric?
- Does the sampler preserve constraints such as graph validity, chirality, coordinate frame, or sequence length?

## Related

- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/latent-variable-model|Latent variable model]]
- [[concepts/generative-models/guidance|Guidance]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
- [[concepts/llm/decoding|Decoding]]
