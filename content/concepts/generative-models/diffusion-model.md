---
title: Diffusion Model
tags:
  - diffusion-model
  - generative-model
  - machine-learning
---

# Diffusion Model

A diffusion model learns to reverse a gradual noising process: it is trained to denoise corrupted data and generates samples by iteratively denoising from noise.

A common forward noising form is:

$$
x_t = \sqrt{\bar{\alpha}_t}x_0
+ \sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I)
$$

The model is often trained to predict $\epsilon$, $x_0$, or a related velocity target.

## Forward Process

In discrete-time DDPM notation:

$$
q(x_t\mid x_{t-1})
=
\mathcal{N}
\left(
\sqrt{\alpha_t}x_{t-1},
(1-\alpha_t)I
\right)
$$

and the closed-form marginal is:

$$
q(x_t\mid x_0)
=
\mathcal{N}
\left(
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right)
$$

where:

$$
\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s
$$

This marginal is what allows training with randomly sampled timesteps instead of simulating every forward step.

A common noise-prediction objective is:

$$
\mathcal{L}_{\epsilon}
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\left\|
\epsilon
- \epsilon_\theta(x_t,t)
\right\|_2^2
\right]
$$

The reverse process samples from noise by repeatedly applying a learned denoising transition:

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}
\left(
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right)
$$

Alternative parameterizations predict $x_0$ or velocity $v$:

$$
v = \alpha_t \epsilon - \sigma_t x_0
$$

The parameterization affects stability, guidance behavior, and solver design.

## Parameterization Choices

| Target | Model Predicts | Useful When | Caveat |
| --- | --- | --- | --- |
| noise $\epsilon$ | $\epsilon_\theta(x_t,t,c)$ | classic DDPM-style training | sample quality depends on noise schedule and solver |
| clean data $x_0$ | $\hat{x}_{0,\theta}(x_t,t,c)$ | interpretable reconstruction target | can be unstable at high noise |
| score | $\nabla_{x_t}\log p_t(x_t)$ | score-based SDE/ODE view | scale depends on noise convention |
| velocity $v$ | $v_\theta(x_t,t,c)$ | diffusion/flow bridge and stable interpolation | must define $\alpha_t,\sigma_t$ convention |

For the common parameterization

$$
x_t = \alpha_t x_0 + \sigma_t \epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I),
$$

a velocity target is often written:

$$
v_t = \alpha_t \epsilon - \sigma_t x_0.
$$

Always record which target a paper uses before comparing losses or samplers.

The score associated with the same Gaussian convention is:

$$
s_\theta(x_t,t)
\approx
-\frac{\epsilon_\theta(x_t,t)}{\sigma_t}
$$

with $\sigma_t^2=1-\bar{\alpha}_t$ in the simple variance-preserving notation. This connects diffusion noise prediction to [[concepts/generative-models/score-based-model|score-based models]].

## Conditioning

Conditional diffusion adds context $c$:

$$
\epsilon_\theta(x_t,t,c)
$$

Guidance changes the effective score or denoising direction, which can improve fidelity but reduce diversity.

Classifier-free guidance often combines conditional and unconditional predictions:

$$
\epsilon_{\mathrm{guided}}
=
\epsilon_\theta(x_t,t,\varnothing)
+
w\left[
\epsilon_\theta(x_t,t,c)
-
\epsilon_\theta(x_t,t,\varnothing)
\right]
$$

where $w$ is the guidance scale. Larger $w$ can improve condition satisfaction while reducing diversity or validity.

## Sampling Budget

Diffusion quality is tied to solver and step count:

$$
x_T \sim \mathcal{N}(0,I),
\qquad
x_{t-1} = \operatorname{Step}_\theta(x_t,t,c;\eta)
$$

where $\eta$ represents sampler settings such as stochasticity, guidance scale, schedule, and step size. A benchmark should report these settings because a model can look better simply by spending more sampling steps or stronger guidance.

## Sampler Families

| Sampler | Shape | Claim Affected |
| --- | --- | --- |
| ancestral DDPM | stochastic reverse chain | diversity and baseline comparability |
| DDIM-style | deterministic or partially stochastic trajectory | fewer steps, different diversity |
| SDE sampler | stochastic continuous-time reverse process | score quality and noise schedule |
| probability-flow ODE | deterministic score-derived ODE | likelihood route and solver budget |
| distilled/consistency sampler | one or few steps | speed at possible quality loss |

NFE, wall time, memory, and invalid-sample denominator should be reported with sample quality.

## Evaluation Boundary

| Claim | Required Evidence |
| --- | --- |
| better likelihood or denoising | NLL/ELBO or denoising loss under matched schedule |
| better sample quality | task-specific sample metrics and human/domain checks |
| better conditional control | condition satisfaction measured separately from diversity |
| faster generation | quality at matched NFE, wall time, memory, and hardware |
| better molecular/protein samples | validity, novelty, diversity, constraints, and downstream utility |

## Why It Matters

- Strong sample quality and stable training across images, molecules, and proteins.
- Flexible conditioning enables guided and inverse-problem generation.
- Iterative sampling is the main cost, motivating faster samplers.

## Failure Modes

- Too few sampling steps can produce invalid or low-detail samples.
- Strong guidance can collapse diversity or overfit the condition.
- The noise schedule may be poorly matched to data scale or geometry.
- Evaluation can confuse validity, novelty, diversity, and downstream utility.

## Checks

- Does the noise schedule fit the data scale?
- Is conditioning/guidance trading diversity for fidelity?
- How many sampling steps are needed for acceptable quality?
- Are sampling steps and guidance scales fixed across comparisons?
- Is the prediction target $\epsilon$, $x_0$, score, or velocity clearly stated?
- Are NFE, solver, stochasticity, and guidance scale matched across baselines?
- Are validity, diversity, novelty, and task utility reported separately?
- Is the noising convention variance-preserving, variance-exploding, or another schedule?
- Are filtered or repaired samples counted in the denominator?

## Related

- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/guidance|Guidance]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/consistency-model|Consistency model]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
