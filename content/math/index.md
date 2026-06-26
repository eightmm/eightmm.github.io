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

이 블로그에서 Math는 독립적인 수학 교과서가 아니라, AI 문서의 수식을 읽기 위한 최소 공통 언어입니다. 더 깊은 증명보다 "이 식이 어떤 object, distribution, optimization target을 말하는가"를 우선합니다.

## Core Foundations

- [[concepts/math/index|Math foundations]]: canonical wiki index
- Linear algebra: [[concepts/math/linear-algebra|Linear algebra]], [[concepts/math/vector-norm-similarity|Vector norm and similarity]], [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]], [[concepts/math/singular-value-decomposition|Singular value decomposition]]
- Calculus and optimization: [[concepts/math/calculus|Calculus]], [[concepts/math/matrix-calculus|Matrix calculus]], [[concepts/math/jacobian-hessian|Jacobian and Hessian]]
- Probability: [[concepts/math/random-variable|Random variable]], [[concepts/math/probability-distribution|Probability distribution]], [[concepts/math/normal-distribution|Normal distribution]], [[concepts/math/expectation|Expectation]], [[concepts/math/bayes-rule|Bayes rule]]
- Statistics: [[concepts/math/statistical-estimator|Statistical estimator]], [[concepts/math/covariance-correlation|Covariance and correlation]], [[concepts/math/central-limit-theorem|Central limit theorem]], [[concepts/math/hypothesis-testing|Hypothesis testing]], [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]]
- Estimation and likelihood: [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]], [[concepts/math/maximum-likelihood|Maximum likelihood]], [[concepts/math/entropy-kl|Entropy and KL divergence]]
- Geometry and symmetry: [[concepts/math/geometry|Geometry]], [[concepts/math/symmetry-group|Symmetry group]]

## Formula Checklist

수식이 나오면 먼저 아래를 확인합니다.

- Object: scalar, vector, matrix, sequence, graph, coordinate, distribution 중 무엇인가?
- Index: batch index, token index, residue index, graph node index, time index가 무엇인가?
- Distribution: expectation이 data, model, noise, policy, test distribution 중 어디에서 잡히는가?
- Parameter: 미분 대상이 input $x$, parameter $\theta$, coordinate $X$, time $t$ 중 무엇인가?
- Objective: 식이 loss, likelihood, score, reward, metric, constraint 중 무엇인가?
- Symmetry: permutation, translation, rotation, scale 변환에서 보존돼야 할 양이 무엇인가?
- Estimation: population quantity와 finite-sample estimate가 구분되어 있는가?

## Where It Connects

- Architecture: [[concepts/architectures/linear-layer|Linear layer]], [[concepts/architectures/attention|Attention]], [[concepts/architectures/normalization|Normalization]]
- Learning: [[concepts/math/calculus|Calculus]], [[concepts/math/matrix-calculus|Matrix calculus]], [[concepts/machine-learning/empirical-risk-minimization|Empirical risk minimization]], [[concepts/machine-learning/loss-function|Loss function]], [[concepts/machine-learning/stochastic-gradient|Stochastic gradient]], [[concepts/machine-learning/optimization|Optimization]], [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- Generation: [[concepts/math/probability-distribution|Probability distribution]], [[concepts/math/maximum-likelihood|Maximum likelihood]], [[concepts/math/entropy-kl|Entropy and KL divergence]], [[concepts/generative-models/diffusion-model|Diffusion model]], [[concepts/generative-models/flow-matching|Flow matching]], [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- Geometry: [[concepts/geometric-deep-learning/index|Geometric deep learning]], [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]], [[concepts/geometric-deep-learning/distance-geometry|Distance geometry]], [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- Evaluation: [[concepts/evaluation/metric|Metric]], [[concepts/evaluation/confidence-interval|Confidence interval]], [[concepts/evaluation/statistical-significance|Statistical significance]]

## Reading Path

1. Start with [[concepts/math/linear-algebra|Linear algebra]] and [[concepts/math/vector-norm-similarity|Vector norm and similarity]] for vectors, matrices, projections, distances, and embedding scores.
2. Use [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]] and [[concepts/math/singular-value-decomposition|Singular value decomposition]] for PCA, low-rank structure, stability, and representation analysis.
3. Use [[concepts/math/calculus|Calculus]], [[concepts/math/matrix-calculus|Matrix calculus]], and [[concepts/math/jacobian-hessian|Jacobian and Hessian]] for gradients, curvature, and optimization.
4. Use [[concepts/machine-learning/empirical-risk-minimization|Empirical risk minimization]], [[concepts/machine-learning/stochastic-gradient|stochastic gradient]], and [[concepts/machine-learning/gradient-descent|gradient descent]] to connect formulas to training.
5. Use [[concepts/math/random-variable|Random variable]], [[concepts/math/probability-distribution|Probability distribution]], [[concepts/math/expectation|Expectation]], and [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]] for likelihood, loss, sampling, and estimates.
6. Use [[concepts/math/covariance-correlation|Covariance and correlation]], [[concepts/math/normal-distribution|Normal distribution]], [[concepts/math/central-limit-theorem|Central limit theorem]], [[concepts/math/statistical-estimator|Statistical estimator]], [[concepts/math/hypothesis-testing|Hypothesis testing]], and [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]] when interpreting experiments.
7. Use [[concepts/math/maximum-likelihood|Maximum likelihood]] and [[concepts/math/entropy-kl|Entropy and KL divergence]] for generative models and representation learning.
8. Use [[concepts/math/geometry|Geometry]] and [[concepts/math/symmetry-group|Symmetry group]] for graph, structure, molecule, and protein notes.
9. Use [[concepts/evaluation/confidence-interval|Confidence interval]] and [[concepts/evaluation/statistical-significance|Statistical significance]] when interpreting reported results.

## AI 연결 예시

- Linear layer는 matrix multiplication입니다: [[concepts/architectures/linear-layer|Linear layer]]
- Attention은 dot product, softmax, weighted sum입니다: [[concepts/architectures/attention|Attention]]
- Loss는 probability model이나 decision objective를 반영합니다: [[concepts/machine-learning/loss-function|Loss function]]
- Diffusion/flow는 probability path와 vector field를 다룹니다: [[concepts/generative-models/diffusion-model|Diffusion model]], [[concepts/generative-models/flow-matching|Flow matching]]
- Equivariance는 group action 아래에서 output이 어떻게 변해야 하는지를 말합니다: [[concepts/geometric-deep-learning/equivariance|Equivariance]]

## Related

- [[ai/index|AI]]
- [[bio-ai/index|Bio-AI]]
- [[concepts/index|Concepts]]
- [[papers/index|Papers]]
