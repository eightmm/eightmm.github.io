---
title: Math
tags:
  - math
---

# Math

수학은 AI, Computational Biology, geometry, generative model, evaluation을 읽기 위한 공통 기반입니다. 이 섹션은 “수학을 깊게 증명하는 곳”이 아니라, 논문과 구현에서 나오는 식을 해석하기 위한 언어를 정리하는 곳입니다.

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

## 핵심 기반

| 영역 | 용도 | 시작점 |
| --- | --- | --- |
| Foundations | definitions, formulas, reusable checks | [Math foundations](/concepts/math) |
| Formula reading | paper equations, shapes, distributions, objectives, estimators, metrics | [Formula Reading](/math/formula-reading) |
| Linear algebra | vectors, matrices, tensor shapes, embeddings, projections, similarity, SVD | [Linear algebra](/math/linear-algebra), [Tensor shape notation](/concepts/math/tensor-shape-notation) |
| Calculus and optimization | derivatives, chain rule, Jacobians, Hessians, constraints, backpropagation math | [Calculus and gradients](/math/calculus-gradients), [Chain rule](/concepts/math/chain-rule), [Constrained optimization](/concepts/math/constrained-optimization) |
| Probability | random variables, distributions, estimators, Bayesian inference, uncertainty, hypothesis tests | [Probability and statistics](/math/probability-statistics) |
| Information | maximum likelihood, entropy, KL, cross-entropy, generative objectives | [Information and likelihood](/math/information-likelihood) |
| Geometry | groups, frames, invariance, equivariance, coordinate modeling | [Geometry and symmetry](/math/geometry-symmetry), [Geometry](/concepts/math/geometry), [Symmetry group](/concepts/math/symmetry-group) |
| Graphs | sets, neighborhoods, permutations, graph modeling | [Discrete math and graphs](/math/discrete-graphs) |
| Dynamics | discrete updates, ODEs, SDEs, vector fields, generative flows, molecular dynamics | [Dynamical systems](/math/dynamical-systems) |
| Numerics | floating point, stable softmax, log-sum-exp, conditioning, precision | [Numerical computing](/math/numerical-computing) |
| Evaluation | metrics, confidence intervals, calibration, statistical comparison | [Evaluation math](/math/evaluation-math) |

## 수식 체크리스트

수식이 나오면 먼저 아래를 확인합니다.

- Object: scalar, vector, matrix, sequence, graph, coordinate, distribution 중 무엇인가?
- Index: batch index, token index, residue index, graph node index, time index가 무엇인가?
- Distribution: expectation이 data, model, noise, policy, test distribution 중 어디에서 잡히는가?
- Parameter: 미분 대상이 input $x$, parameter $\theta$, coordinate $X$, time $t$ 중 무엇인가?
- Objective: 식이 loss, likelihood, score, reward, metric, constraint 중 무엇인가?
- Symmetry: permutation, translation, rotation, scale 변환에서 보존돼야 할 양이 무엇인가?
- Estimation: population quantity와 finite-sample estimate가 구분되어 있는가?
- Depth: post, concept note, paper note 중 어느 수준까지 풀어써야 하는가?

## 기본 수식 읽기 순서

AI 수식은 아래 순서로 읽으면 대부분 정리됩니다.

1. Type: scalar, vector, matrix, tensor, graph, distribution 중 무엇인가?
2. Shape: 각 축이 batch, token, node, coordinate, feature, time 중 무엇인가?
3. Domain: 변수의 가능한 값과 constraint가 무엇인가?
4. Operation: linear map, normalization, expectation, derivative, sampling, aggregation 중 무엇인가?
5. Target: loss, metric, likelihood, score, energy, reward, constraint 중 무엇인가?
6. Estimate: population quantity인지 finite sample estimate인지 구분합니다.

## Formula by Claim Type

| Claim | Route | Check |
| --- | --- | --- |
| Architecture formula | [Linear layer](/concepts/architectures/linear-layer), [Attention](/concepts/architectures/attention) | shape, axis, projection, normalization |
| Learning objective | [Learning Methods](/ai/learning-methods), [Objective taxonomy](/concepts/learning/objective-taxonomy) | label, mask, contrast, likelihood, denoising, reward |
| Geometry rule | [Geometry and Symmetry](/math/geometry-symmetry), [Equivariance](/concepts/geometric-deep-learning/equivariance) | invariant target인지 equivariant target인지 |
| Evaluation metric | [Evaluation Math](/math/evaluation-math), [Estimator vs metric](/concepts/math/estimator-vs-metric) | population quantity, finite estimate, selection rule |

## 연결되는 곳

| 맥락 | 링크 |
| --- | --- |
| Architecture | [Linear layer](/concepts/architectures/linear-layer), [Attention](/concepts/architectures/attention), [Normalization](/concepts/architectures/normalization) |
| Learning | [Calculus](/concepts/math/calculus), [Matrix calculus](/concepts/math/matrix-calculus), [Constrained optimization](/concepts/math/constrained-optimization), [ERM](/concepts/machine-learning/empirical-risk-minimization), [Loss function](/concepts/machine-learning/loss-function), [Optimization](/concepts/machine-learning/optimization) |
| Objective and metric | [Objective-metric alignment](/concepts/machine-learning/objective-metric-alignment), [Loss function](/concepts/machine-learning/loss-function), [Metric selection](/concepts/evaluation/metric-selection), [Claim-evidence boundary](/concepts/evaluation/claim-evidence-boundary) |
| Selection and uncertainty | [Model selection](/concepts/machine-learning/model-selection), [Confidence interval](/concepts/evaluation/confidence-interval), [Cross-validation](/concepts/evaluation/cross-validation), [Calibration](/concepts/evaluation/calibration) |
| Numerical stability | [Numerical computing](/math/numerical-computing), [Softmax](/concepts/architectures/softmax), [Training stability](/concepts/machine-learning/training-stability), [Memory-compute tradeoff](/concepts/systems/memory-compute-tradeoff) |
| Generation | [Probability distribution](/concepts/math/probability-distribution), [Maximum likelihood](/concepts/math/maximum-likelihood), [Entropy and KL](/concepts/math/entropy-kl), [Diffusion](/concepts/generative-models/diffusion-model), [Flow matching](/concepts/generative-models/flow-matching) |
| Geometry | [Geometric deep learning](/concepts/geometric-deep-learning), [Coordinate frame](/concepts/geometric-deep-learning/coordinate-frame), [Distance geometry](/concepts/geometric-deep-learning/distance-geometry), [Equivariance](/concepts/geometric-deep-learning/equivariance), [Coordinate update](/concepts/geometric-deep-learning/coordinate-update) |
| Graphs and dynamics | [Discrete math and graphs](/math/discrete-graphs), [Dynamical systems](/math/dynamical-systems), [GNN](/concepts/architectures/gnn), [Graph construction](/concepts/architectures/graph-construction), [Probability flow ODE](/concepts/generative-models/probability-flow-ode) |
| Evaluation | [Metric](/concepts/evaluation/metric), [Confidence interval](/concepts/evaluation/confidence-interval), [Statistical significance](/concepts/evaluation/statistical-significance) |
| Claim boundary | [Claim-evidence boundary](/concepts/evaluation/claim-evidence-boundary), [Benchmark intake](/concepts/data/benchmark-intake) |

## 읽기 경로

1. vector, matrix, axis, projection, distance, embedding score를 읽으려면 [[concepts/math/linear-algebra|Linear algebra]], [[concepts/math/tensor-shape-notation|Tensor shape notation]], [[concepts/math/vector-norm-similarity|Vector norm and similarity]]에서 시작합니다.
2. PCA, low-rank structure, stability, representation analysis에는 [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]와 [[concepts/math/singular-value-decomposition|Singular value decomposition]]를 봅니다.
3. gradient, curvature, backpropagation math에는 [[concepts/math/calculus|Calculus]], [[concepts/math/matrix-calculus|Matrix calculus]], [[concepts/math/jacobian-hessian|Jacobian and Hessian]]를 봅니다.
4. 수식을 AI optimization과 연결할 때는 [[concepts/machine-learning/empirical-risk-minimization|Empirical risk minimization]], [[concepts/machine-learning/stochastic-gradient|stochastic gradient]], [[concepts/machine-learning/gradient-descent|gradient descent]]를 봅니다.
5. likelihood, loss, sampling, estimate에는 [[concepts/math/random-variable|Random variable]], [[concepts/math/probability-distribution|Probability distribution]], [[concepts/math/expectation|Expectation]], [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]]를 봅니다.
6. 실험 결과 해석에는 [[concepts/math/covariance-correlation|Covariance and correlation]], [[concepts/math/normal-distribution|Normal distribution]], [[concepts/math/central-limit-theorem|Central limit theorem]], [[concepts/math/statistical-estimator|Statistical estimator]], [[concepts/math/bayesian-inference|Bayesian inference]], [[concepts/math/hypothesis-testing|Hypothesis testing]], [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]]를 봅니다.
7. generative model과 representation learning에는 [[concepts/math/maximum-likelihood|Maximum likelihood]]와 [[concepts/math/entropy-kl|Entropy and KL divergence]]를 봅니다.
8. graph, structure, molecule, protein note에는 [[concepts/math/geometry|Geometry]]와 [[concepts/math/symmetry-group|Symmetry group]]를 봅니다.
9. graph neural network, molecular graph, contact map, retrieval, set function에는 [[math/discrete-graphs|Discrete math and graphs]]를 봅니다.
10. diffusion, flow matching, probability-flow ODE, recurrent model, molecular dynamics, iterative refinement에는 [[math/dynamical-systems|Dynamical systems]]를 봅니다.
11. exponential, probability, mixed precision, reduction, NaN이 나오면 [[math/numerical-computing|Numerical computing]]을 봅니다.
12. 보고된 결과를 해석할 때는 [[concepts/evaluation/confidence-interval|Confidence interval]]과 [[concepts/evaluation/statistical-significance|Statistical significance]]를 봅니다.

## 경계

- Math는 derivative, probability, likelihood, geometry, metric, numerical stability처럼 object와 formula를 설명합니다.
- AI는 loss, optimizer, schedule, architecture, learning method처럼 modeling과 training 선택을 설명합니다.
- Computational Biology는 protein, ligand, pocket, sequence, structure, assay, split처럼 domain object와 workflow를 설명합니다.
- Infra는 GPU memory, Slurm, storage, reproducibility, server operation처럼 resource behavior를 설명합니다.

## AI 연결 예시

- Linear layer는 matrix multiplication입니다: [[concepts/architectures/linear-layer|Linear layer]]
- Attention은 dot product, softmax, weighted sum입니다: [[concepts/architectures/attention|Attention]]
- Loss는 probability model이나 decision objective를 반영합니다: [[concepts/machine-learning/loss-function|Loss function]]
- Diffusion/flow는 probability path와 vector field를 다룹니다: [[concepts/generative-models/diffusion-model|Diffusion model]], [[concepts/generative-models/flow-matching|Flow matching]]
- Equivariance는 group action 아래에서 output이 어떻게 변해야 하는지를 말합니다: [[concepts/geometric-deep-learning/equivariance|Equivariance]]

## 논문 수식을 읽을 때

새 논문의 수식은 멋있어 보이는 notation보다 어떤 양을 정의하는지 먼저 봅니다.

| 수식 종류 | 먼저 확인할 것 | 시작점 |
| --- | --- | --- |
| Linear operation | shape, axis, projection, rank, similarity | [Linear algebra](/math/linear-algebra), [Tensor shape notation](/concepts/math/tensor-shape-notation) |
| Gradient or update | objective, parameter, gradient estimate, optimizer state | [Calculus and gradients](/math/calculus-gradients) |
| Constraint | feasible set, Lagrangian, penalty, projection, invalid-output handling | [Constrained optimization](/concepts/math/constrained-optimization) |
| Probability / expectation | random variable, conditioning, sampling distribution | [Probability and statistics](/math/probability-statistics) |
| Likelihood / entropy / KL | modeled distribution, target distribution, bound or approximation | [Information and likelihood](/math/information-likelihood) |
| Symmetry / coordinate rule | transformation group, invariant target, equivariant target | [Geometry and symmetry](/math/geometry-symmetry) |
| Graph or set equation | node/edge/set unit, permutation behavior, aggregation | [Discrete math and graphs](/math/discrete-graphs) |
| ODE/SDE/flow | state, time, vector field, integration or sampling path | [Dynamical systems](/math/dynamical-systems) |
| Metric / comparison | point estimate, uncertainty, paired examples, selection rule | [Evaluation math](/math/evaluation-math) |

## Related

- [[ai/index|AI]]
- [[math/formula-reading|Formula Reading]]
- [[molecular-modeling/index|Computational Biology]]
- [[concepts/index|Concepts]]
- [[papers/index|Papers]]
