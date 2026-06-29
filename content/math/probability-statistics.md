---
title: Probability and Statistics
tags:
  - math
  - probability
  - statistics
---

# Probability and Statistics

Probability는 uncertainty와 data-generating process를 설명합니다. Statistics는 finite sample로 unknown quantity를 추정하는 방법을 설명합니다.

$$
\mathbb{E}_{(x,y)\sim p_{\mathrm{data}}}
[\mathcal{L}(f(x), y)]
$$

Expectation 아래의 distribution은 loss 자체만큼 중요합니다.

## Route Map

| Route | Use for | Start |
| --- | --- | --- |
| Random objects | 어떤 variable, event, sample space, distribution을 모델링하는가 | [Random variable](/concepts/math/random-variable), [Probability distribution](/concepts/math/probability-distribution) |
| Expectations | loss, risk, metric, Monte Carlo estimate, sampling average | [Expectation](/concepts/math/expectation), [Monte Carlo estimation](/concepts/math/monte-carlo-estimation) |
| Common distributions | noise model, uncertainty assumption, approximate sampling law | [Normal distribution](/concepts/math/normal-distribution) |
| Conditioning and inference | posterior, likelihood, prior, Bayesian update | [Bayes rule](/concepts/math/bayes-rule), [Bayesian inference](/concepts/math/bayesian-inference) |
| Likelihood and information | NLL, cross-entropy, KL, generative objectives | [Maximum likelihood](/concepts/math/maximum-likelihood), [Entropy and KL divergence](/concepts/math/entropy-kl), [Information and likelihood](/math/information-likelihood) |
| Statistical evidence | finite-sample estimates, uncertainty, hypothesis tests, bias/variance | [Statistical estimator](/concepts/math/statistical-estimator), [Hypothesis testing](/concepts/math/hypothesis-testing), [Bias-variance tradeoff](/concepts/math/bias-variance-tradeoff) |

## Distribution Map

| Quantity | Meaning | Common AI use |
| --- | --- | --- |
| $p(x)$ | input의 marginal distribution | data modeling, density estimation |
| $p(y\mid x)$ | conditional label distribution | classification, regression uncertainty |
| $p_\theta(y\mid x)$ | model-predicted conditional distribution | probabilistic prediction과 calibration |
| $p_{\mathrm{train}}(x,y)$ | training distribution | empirical risk minimization |
| $p_{\mathrm{test}}(x,y)$ | evaluation distribution | generalization claim |
| $q(z\mid x)$ | approximate posterior 또는 encoder distribution | VAE, latent-variable inference |
| $p(z)$ | latent variable의 prior | generative modeling과 regularization |
| $p_\theta(x\mid z)$ | likelihood 또는 decoder distribution | reconstruction과 conditional generation |

## Probability Reading Pattern

확률식은 아래 질문으로 읽습니다.

| Question | Example |
| --- | --- |
| 무엇이 random인가? | data example, label, latent variable, noise, model sample |
| 무엇에 condition하는가? | $p(y\mid x)$, $p(x\mid c)$, $p(z\mid x)$ |
| 무엇을 learn하는가? | parameterized distribution $p_\theta$ 또는 encoder $q_\phi$ |
| 무엇을 estimate하는가? | expectation, risk, likelihood, metric, posterior |
| sample source는 무엇인가? | train distribution, test distribution, deployment distribution |

Bayes rule connects posterior, likelihood, and prior:

$$
p(y\mid x)
=
\frac{p(x\mid y)p(y)}{p(x)}
$$

이 식은 probabilistic classifier, latent-variable model, uncertainty note를 읽을 때 유용합니다.

## Conditional Modeling

대부분의 supervised AI claim은 conditional distribution으로 쓸 수 있습니다.

$$
p_\theta(y\mid x)
$$

Training은 observed label이 model 아래에서 likely하도록 parameter를 선택합니다.

$$
\theta^\star
=
\operatorname*{arg\,max}_\theta
\sum_{i=1}^{n}
\log p_\theta(y_i\mid x_i)
$$

Logit $z=f_\theta(x)$를 쓰는 classification에서는:

$$
p_\theta(y=k\mid x)
=
\frac{\exp z_k}{\sum_{j=1}^{K}\exp z_j}
$$

한 example에 대한 negative log-likelihood는 아래와 같습니다.

$$
\mathcal{L}_{\mathrm{CE}}
=
-\log p_\theta(y\mid x)
$$

그래서 cross-entropy, maximum likelihood, calibration, uncertainty는 서로 떨어진 trick이 아니라 하나로 연결된 topic입니다.

## Distribution Shift

Generalization claim은 expectation이 어떤 distribution을 가리키는지에 의존합니다.

$$
R_{\mathrm{deploy}}(\theta)
=
\mathbb{E}_{(x,y)\sim p_{\mathrm{deploy}}}
\left[\mathcal{L}(f_\theta(x),y)\right]
$$

Empirical test estimate가 deployment evidence가 되려면 아래가 필요합니다.

$$
p_{\mathrm{test}}(x,y)
\approx
p_{\mathrm{deploy}}(x,y)
$$

Computational biology에서는 scaffold shift, protein-family shift, assay/source shift, structure-source shift, negative-set construction 때문에 이 조건이 깨질 수 있습니다.

## Statistics

Statistics는 finite observation을 population quantity에 대한 claim으로 바꿉니다. Correlation, uncertainty, hypothesis test, benchmark estimate는 exact fact가 아니라 assumption이 붙은 estimate로 읽어야 합니다.

| Topic | Start |
| --- | --- |
| Covariation | [Covariance and correlation](/concepts/math/covariance-correlation) |
| Asymptotic approximation | [Central limit theorem](/concepts/math/central-limit-theorem) |
| Evidence against a null | [Hypothesis testing](/concepts/math/hypothesis-testing) |
| Error decomposition | [Bias-variance tradeoff](/concepts/math/bias-variance-tradeoff) |
| Sampling estimate | [Monte Carlo estimation](/concepts/math/monte-carlo-estimation) |

## Estimation

A finite dataset gives an estimate of a population quantity:

$$
\mu = \mathbb{E}_{X\sim p}[h(X)],
\qquad
\hat{\mu}
=
\frac{1}{n}\sum_{i=1}^{n} h(x_i)
$$

Error는 sampling noise, bias, sample 사이의 dependence, sample distribution과 target distribution의 일치 여부에 의해 좌우됩니다.

For an estimator $\hat{\theta}$:

$$
\operatorname{MSE}(\hat{\theta})
=
\operatorname{Var}(\hat{\theta})
+
\operatorname{Bias}(\hat{\theta})^2
$$

## AI Connections

- Probabilistic prediction에는 score만이 아니라 calibrated probability가 필요합니다.
- Dataset shift는 expected risk 아래의 distribution을 바꿉니다.
- Uncertainty estimation은 어떤 randomness를 모델링하는지에 의존합니다.
- Bayesian inference는 prior, likelihood, posterior, posterior predictive claim을 분리합니다.
- Hypothesis testing과 confidence interval은 benchmark difference 해석을 돕습니다.

## Computational Biology Connections

| Question | Statistical object | Common risk |
| --- | --- | --- |
| activity label이 reliable한가? | assay endpoint의 noisy observation | assay protocol과 unit difference |
| split이 transfer를 test하는가? | target distribution 아래의 estimate | scaffold, family, source leakage |
| screening score가 유용한가? | candidate pool 위의 ranking statistic | active prevalence와 decoy bias |
| uncertainty가 의미 있는가? | predictive distribution 또는 interval | wrong domain에서 calibration됨 |
| generative model이 valid한가? | sample distribution $p_\theta(x)$ | utility나 novelty 없는 validity |

## Checks

- probability가 conditional인가 marginal인가?
- estimate가 biased, high-variance, data-leaking 중 어디에 해당하는가?
- test distribution이 deployment distribution과 같은가?
- repeated evaluation이 multiple-comparison risk를 만들고 있는가?
- claim이 likelihood, decision quality, ranking, calibration, downstream utility 중 무엇에 관한 것인가?
- reported metric이 text가 주장하는 expectation과 같은 대상을 estimate하는가?

## Related

- [[math/index|Math]]
- [[ai/evaluation|Evaluation]]
- [[ai/machine-learning|Machine learning]]
- [[molecular-modeling/data-evaluation|Computational Biology data and evaluation]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/data/dataset-shift|Dataset shift]]
- [[concepts/math/bayesian-inference|Bayesian inference]]
