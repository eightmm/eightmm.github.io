---
title: Dynamical Systems
tags:
  - math
  - dynamical-systems
  - generative-models
---

# Dynamical Systems

Dynamical system은 state가 시간 또는 연속 경로를 따라 어떻게 변하는지 설명합니다. AI에서는 recurrent model, residual network, diffusion model, flow matching, probability-flow ODE, control, reinforcement learning, molecular dynamics에서 반복적으로 등장합니다.

## Route Map

| 질문 | 시작점 | 쓰임 |
| --- | --- | --- |
| state가 step-by-step으로 update되는가? | [RNN](/concepts/architectures/rnn), [Residual network](/concepts/architectures/residual-network) | recurrence, iterative refinement, deep residual block |
| model이 continuous flow인가? | [Flow matching](/concepts/generative-models/flow-matching), [Probability flow ODE](/concepts/generative-models/probability-flow-ode), [Rectified flow](/concepts/generative-models/rectified-flow) | sample transport, velocity field |
| noise가 dynamics의 일부인가? | [Diffusion model](/concepts/generative-models/diffusion-model) | stochastic path, denoising objective |
| agent 또는 controller가 시간에 따라 행동하는가? | [Reinforcement learning](/concepts/learning/reinforcement-learning) | state, action, reward, policy dynamics |
| state가 물리적 상태인가? | [Molecular dynamics](/concepts/molecular-modeling/molecular-dynamics) | coordinate와 force evolution |

## Discrete-Time Dynamics

Discrete-time system은 state를 step 단위로 update합니다.

$$
x_{t+1}
=
F_\theta(x_t,t)
$$

이 관점은 recurrent network, iterative refinement, optimization step, 일부 agent state transition을 설명합니다.

## Continuous-Time Dynamics

Continuous-time system은 보통 ODE로 씁니다.

$$
\frac{dx(t)}{dt}
=
f_\theta(x(t),t)
$$

$f_\theta$는 vector field입니다. 각 위치에서 이동 방향과 속도를 알려줍니다.

## Flow Map

시간 $s$에서 $t$까지 ODE를 풀면 flow map을 정의할 수 있습니다.

$$
x(t)
=
\Phi_{s\to t}(x(s))
$$

Generative flow는 이 관점을 사용해 simple distribution의 sample을 data distribution으로 이동시킵니다.

## Stochastic Dynamics

Diffusion 계열 model은 보통 noise를 포함합니다. 단순화한 stochastic differential equation은 다음과 같습니다.

$$
dx
=
f(x,t)\,dt
+
g(t)\,dW_t
$$

$W_t$는 Brownian motion입니다. Drift $f$는 deterministic movement를, diffusion term $g(t)$는 noise scale을 조절합니다.

## Residual Networks as Dynamics

A residual block:

$$
h_{l+1}
=
h_l + F_\theta(h_l)
$$

이는 continuous dynamics의 discrete approximation으로 읽을 수 있습니다.

$$
\frac{dh(t)}{dt}
\approx
F_\theta(h(t))
$$

이 관점은 deep architecture, stability, continuous generative model을 연결해 줍니다.

## Molecular Dynamics View

Molecular dynamics는 coordinate 위의 physical dynamical system입니다.

$$
m_i\frac{d^2x_i}{dt^2}
=
-\nabla_{x_i}E(X)
$$

State에는 position과 velocity가 포함될 수 있습니다. AI note에서는 dynamics가 fixed physical model로 simulate되는지, neural model로 learned되는지, 아니면 post hoc analysis로만 쓰이는지를 구분하는 것이 중요합니다.

## Checks

- time이 discrete, continuous, learned, 또는 단순 ordering variable 중 무엇인가?
- state가 vector, sequence, graph, coordinate set, distribution 중 무엇인가?
- dynamics가 deterministic인가 stochastic인가?
- model이 score, vector field, transition kernel, policy 중 무엇을 학습하는가?
- numerical integration이 method의 일부인가, 아니면 conceptual analogy인가?
- molecular dynamics라면 force field, time step, initialization, analyzed frame이 명시되어 있는가?

## Related

- [[math/index|Math]]
- [[math/calculus-gradients|Calculus and gradients]]
- [[math/probability-statistics|Probability and statistics]]
- [[math/numerical-computing|Numerical computing]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/molecular-modeling/molecular-dynamics|Molecular dynamics]]
- [[concepts/molecular-modeling/force-field|Force field]]
