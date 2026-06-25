---
title: Math
tags:
  - math
---

# Math

수학은 AI, Bio-AI, geometry, generative model, evaluation을 읽기 위한 공통 기반입니다. 이 페이지는 사이드바와 홈에서 바로 들어오는 한글 gateway이고, 세부 개념은 영어 canonical wiki note로 유지합니다.

공식은 모델을 어렵게 보이게 하려는 장식이 아니라, 어떤 quantity를 예측하고 최적화하고 평가하는지 고정하는 도구입니다.

$$
\text{modeling}
=
\text{representation}
+
\text{objective}
+
\text{evaluation}
$$

## Core Foundations

- [[concepts/math/index|Math foundations]]
- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/math/singular-value-decomposition|Singular value decomposition]]
- [[concepts/math/calculus|Calculus]]
- [[concepts/math/matrix-calculus|Matrix calculus]]
- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/math/random-variable|Random variable]]
- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/math/bayes-rule|Bayes rule]]
- [[concepts/math/statistical-estimator|Statistical estimator]]
- [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]]

## Where It Connects

- Architecture: [[concepts/architectures/linear-layer|Linear layer]], [[concepts/architectures/attention|Attention]], [[concepts/architectures/normalization|Normalization]]
- Learning: [[concepts/math/calculus|Calculus]], [[concepts/math/matrix-calculus|Matrix calculus]], [[concepts/machine-learning/empirical-risk-minimization|Empirical risk minimization]], [[concepts/machine-learning/loss-function|Loss function]], [[concepts/machine-learning/stochastic-gradient|Stochastic gradient]], [[concepts/machine-learning/optimization|Optimization]], [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- Generation: [[concepts/math/probability-distribution|Probability distribution]], [[concepts/math/maximum-likelihood|Maximum likelihood]], [[concepts/math/entropy-kl|Entropy and KL divergence]], [[concepts/generative-models/diffusion-model|Diffusion model]], [[concepts/generative-models/flow-matching|Flow matching]], [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- Geometry: [[concepts/geometric-deep-learning/index|Geometric deep learning]], [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]], [[concepts/geometric-deep-learning/distance-geometry|Distance geometry]], [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- Evaluation: [[concepts/evaluation/metric|Metric]], [[concepts/evaluation/confidence-interval|Confidence interval]], [[concepts/evaluation/statistical-significance|Statistical significance]]

## Reading Path

1. Start with [[concepts/math/linear-algebra|Linear algebra]] for vectors, matrices, projections, and layers.
2. Use [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]] and [[concepts/math/singular-value-decomposition|Singular value decomposition]] for PCA, low-rank structure, stability, and representation analysis.
3. Use [[concepts/math/calculus|Calculus]] and [[concepts/math/matrix-calculus|Matrix calculus]] for gradients, Jacobians, and optimization.
4. Use [[concepts/machine-learning/empirical-risk-minimization|Empirical risk minimization]], [[concepts/machine-learning/stochastic-gradient|stochastic gradient]], and [[concepts/machine-learning/gradient-descent|gradient descent]] to connect formulas to training.
5. Use [[concepts/math/random-variable|Random variable]], [[concepts/math/probability-distribution|Probability distribution]], [[concepts/math/expectation|Expectation]], and [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]] for likelihood, loss, sampling, and estimates.
6. Use [[concepts/math/covariance-correlation|Covariance and correlation]], [[concepts/math/statistical-estimator|Statistical estimator]], and [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]] when interpreting experiments.
7. Use [[concepts/math/maximum-likelihood|Maximum likelihood]] and [[concepts/math/entropy-kl|Entropy and KL divergence]] for generative models and representation learning.
8. Use [[concepts/math/geometry|Geometry]] and [[concepts/math/symmetry-group|Symmetry group]] for graph, structure, molecule, and protein notes.
9. Use [[concepts/evaluation/confidence-interval|Confidence interval]] when interpreting reported results.

## Related

- [[ai/index|AI]]
- [[bio-ai/index|Bio-AI]]
- [[concepts/index|Concepts]]
- [[papers/index|Papers]]
