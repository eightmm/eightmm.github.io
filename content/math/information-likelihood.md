---
title: Information and Likelihood
tags:
  - math
  - likelihood
  - information-theory
---

# Information and Likelihood

Likelihood와 information-theoretic quantity는 probability model, loss function, generative model, representation learning을 연결합니다.

$$
\hat{\theta}
=
\arg\max_\theta
\sum_{i=1}^{n}\log p_\theta(x_i)
$$

Maximum likelihood는 modeling assumption 아래에서 관측된 data에 높은 probability를 주도록 model을 학습합니다.

## Route Map

| 질문 | 시작점 | 쓰임 |
| --- | --- | --- |
| 어떤 probability model을 fit하는가? | [Maximum likelihood](/concepts/math/maximum-likelihood), [Negative log-likelihood](/concepts/machine-learning/negative-log-likelihood) | density estimation, supervised likelihood |
| prior와 evidence를 어떻게 결합하는가? | [Bayesian inference](/concepts/math/bayesian-inference) | posterior reasoning, uncertainty assumption |
| distribution을 어떻게 비교하는가? | [Entropy and KL divergence](/concepts/math/entropy-kl), [Cross-entropy loss](/concepts/machine-learning/cross-entropy-loss) | classification, variational objective, representation learning |
| 보고된 probability가 의미 있는가? | [Probabilistic prediction](/concepts/machine-learning/probabilistic-prediction), [Proper scoring rule](/concepts/evaluation/proper-scoring-rule) | calibration, decision rule, scoring |

## Core Quantities

| Quantity | Formula | 쓰임 |
| --- | --- | --- |
| Negative log-likelihood | $-\log p_\theta(x)$ | 관측 data에 probability model을 fit |
| Cross-entropy | $H(p,q)=-\mathbb{E}_{x\sim p}\log q(x)$ | supervised classification, distribution matching |
| Entropy | $H(p)=-\mathbb{E}_{x\sim p}\log p(x)$ | distribution의 uncertainty |
| KL divergence | $D_{\mathrm{KL}}(p\|q)=\mathbb{E}_{x\sim p}\log \frac{p(x)}{q(x)}$ | distribution을 비대칭적으로 비교 |
| ELBO | $\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))$ | latent-variable model 학습 |

Labeled dataset에서 cross-entropy는 관측 class의 negative log-likelihood입니다.

$$
\mathcal{L}_{\mathrm{CE}}
=
-\sum_{k=1}^{K} y_k \log p_\theta(y=k\mid x)
$$

$y_k$는 one-hot target 또는 soft target입니다.

## Likelihood Reading Checklist

| 질문 | 중요한 이유 | 시작점 |
| --- | --- | --- |
| 어떤 probability가 정의되는가? | $p(x)$, $p(y\mid x)$, $p(x,y)$, $p(x\mid c)$, unnormalized score는 서로 다른 claim을 지지합니다 | [Maximum likelihood](/concepts/math/maximum-likelihood) |
| support가 무엇인가? | probability는 model이 정의한 vocabulary, class set, coordinate space, sample space 안에서만 의미가 있습니다 | [Probability distribution](/concepts/math/probability-distribution) |
| 무엇을 평균내는가? | token, example, graph, atom, residue, pair, trajectory denominator가 objective를 바꿉니다 | [Loss function](/concepts/machine-learning/loss-function) |
| likelihood가 exact인가? | normalized likelihood, variational bound, score objective, implicit model은 같은 evidence가 아닙니다 | [Density estimation](/concepts/machine-learning/density-estimation) |
| loss가 utility와 맞는가? | 더 좋은 likelihood가 더 좋은 ranking, generation, calibration, domain utility를 항상 뜻하지는 않습니다 | [Objective-metric alignment](/concepts/machine-learning/objective-metric-alignment) |
| probability가 어떻게 쓰이는가? | training loss, model selection, calibration, sampling, final decision은 서로 다른 evidence를 요구할 수 있습니다 | [Probabilistic prediction](/concepts/machine-learning/probabilistic-prediction) |

## Generative Models

- [[concepts/generative-models/latent-variable-model|Latent variable model]]
- [[concepts/generative-models/elbo|Evidence lower bound]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]

## Reading Generative Objectives

| Objective Type | 학습 대상 | 볼 점 |
| --- | --- | --- |
| Likelihood | normalized probability 또는 tractable factorization | high likelihood가 useful sample을 항상 뜻하지는 않음 |
| Bayesian posterior | prior와 likelihood를 evidence로 update | posterior approximation과 prior assumption이 중요함 |
| Variational bound | latent encoder와 decoder | bound tightness와 posterior assumption을 확인 |
| Denoising / score | noisy data 위의 gradient 또는 denoising target | sampling path와 noise schedule이 behavior를 정의 |
| Flow matching | probability path 위의 velocity field | path choice와 ODE integration이 sample에 영향 |
| Adversarial | discriminator feedback을 통한 generator | mode collapse와 unstable training을 확인 |

## Checks

- 어떤 distribution을 modeling하는가?
- loss가 likelihood, variational bound, denoising target, score target, velocity target 중 무엇인가?
- probability가 ranking, calibration, sampling, decision-making 중 어디에 쓰이는가?
- 낮은 loss가 해당 task의 downstream utility 개선을 의미하는가?

## Related

- [[math/index|Math]]
- [[ai/generative-models|Generative Models]]
- [[ai/learning-methods|Learning Methods]]
- [[concepts/math/bayesian-inference|Bayesian inference]]
