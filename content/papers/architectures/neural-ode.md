---
title: Neural Ordinary Differential Equations
aliases:
  - papers/neural-ode
  - papers/neural-ordinary-differential-equations
tags:
  - papers
  - architectures
  - continuous-depth
---

# Neural Ordinary Differential Equations

> The paper reframes neural network depth as continuous-time hidden-state dynamics.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Neural Ordinary Differential Equations |
| Authors | Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud |
| Year | 2018 |
| Venue | NeurIPS 2018 |
| arXiv | [1806.07366](https://arxiv.org/abs/1806.07366) |
| Status | full note started |

## One-Line Takeaway

[[papers/architectures/deep-residual-learning|ResNets]] can be read as discrete dynamical systems; Neural ODE asks what happens if the hidden state evolves through a continuous differential equation solved by an ODE solver.

## Question

A residual block updates a hidden state by adding a learned increment:

$$
h_{k+1} = h_k + f_\theta(h_k).
$$

This resembles a forward Euler step for an ordinary differential equation:

$$
h(t+\Delta t)
\approx
h(t) + \Delta t \, f_\theta(h(t),t).
$$

If a ResNet layer stack is a discretized trajectory, then increasing depth is like using more time steps. The paper asks:

> Can we define a neural network by a continuous-time dynamics function and let a numerical ODE solver decide the computation path?

## Main Claim

Instead of specifying a finite list of layers, define the derivative of a hidden state:

$$
\frac{dh(t)}{dt}
=
f_\theta(h(t), t).
$$

Given an initial state $h(t_0)$, the output at time $t_1$ is:

$$
h(t_1)
=
h(t_0)
+
\int_{t_0}^{t_1}
f_\theta(h(t),t)\,dt.
$$

The neural network layer is then:

$$
h(t_1)
=
\operatorname{ODESolve}
\left(
f_\theta, h(t_0), t_0, t_1, \text{tol}
\right).
$$

The depth is no longer an integer layer count. It is the number of function evaluations chosen by the numerical solver.

## Architecture Contract

| Component | Role |
| --- | --- |
| initial state $h(t_0)$ | input representation |
| dynamics function $f_\theta(h,t)$ | neural network defining the derivative |
| ODE solver | integrates the hidden state to target time |
| tolerance | controls numerical accuracy and compute |
| adjoint method | computes gradients through the solve |
| readout | maps final state to task output |

The model changes the usual architecture contract:

$$
\text{layer stack}
\rightarrow
\text{dynamics function plus solver}.
$$

## From ResNet to ODE

A residual network applies:

$$
h_{k+1} = h_k + f_{\theta_k}(h_k).
$$

If all residual blocks share a continuous dynamics view, write:

$$
h_{k+1}
=
h_k + \Delta t \, f_\theta(h_k, t_k).
$$

This is the Euler discretization of:

$$
\frac{dh}{dt}=f_\theta(h,t).
$$

As $\Delta t \rightarrow 0$, the discrete update becomes a continuous trajectory:

$$
h(t_1)
=
h(t_0)
+
\int_{t_0}^{t_1} f_\theta(h(t),t)\,dt.
$$

This is the conceptual bridge:

| ResNet View | Neural ODE View |
| --- | --- |
| finite blocks | continuous trajectory |
| block index $k$ | time $t$ |
| residual function | derivative function |
| fixed depth | adaptive solver steps |
| backprop through layers | adjoint or solver-aware gradients |

## ODE Solver as a Layer

The forward pass is a black-box solve:

$$
z(t_1)
=
\operatorname{ODESolve}(f_\theta, z(t_0), t_0, t_1).
$$

A solver evaluates $f_\theta$ several times:

$$
f_\theta(z(t^{(1)}),t^{(1)}),
f_\theta(z(t^{(2)}),t^{(2)}),
\dots
$$

The number of evaluations is often called NFE:

$$
\operatorname{NFE} = \# \text{ calls to } f_\theta.
$$

NFE is the effective compute measure for a Neural ODE. A model with elegant continuous depth can still be slow if the solver needs many evaluations.

## Adaptive Computation

An adaptive solver chooses step sizes based on estimated numerical error. If the dynamics are simple, it can take large steps. If the dynamics are complex or stiff, it takes smaller steps.

Let the solver target error tolerance be:

$$
\epsilon.
$$

Roughly:

$$
\epsilon \downarrow
\quad \Rightarrow \quad
\text{more function evaluations},
$$

and:

$$
\epsilon \uparrow
\quad \Rightarrow \quad
\text{fewer function evaluations but more numerical error}.
$$

This makes tolerance an architecture and training hyperparameter, not just a numerical detail.

## Adjoint Sensitivity

Naively backpropagating through every solver step can store many intermediate states. The paper uses the adjoint sensitivity method to reduce memory.

Suppose the loss depends on the final state:

$$
L = L(h(t_1)).
$$

Define the adjoint:

$$
a(t) = \frac{\partial L}{\partial h(t)}.
$$

The adjoint evolves backward in time according to:

$$
\frac{da(t)}{dt}
=
-a(t)^\top
\frac{\partial f_\theta(h(t),t)}{\partial h(t)}.
$$

The parameter gradient is:

$$
\frac{dL}{d\theta}
=
-
\int_{t_1}^{t_0}
a(t)^\top
\frac{\partial f_\theta(h(t),t)}{\partial \theta}
\,dt.
$$

In practice, this means solving an augmented ODE backward:

$$
\frac{d}{dt}
\begin{bmatrix}
h(t) \\
a(t) \\
\frac{dL}{d\theta}
\end{bmatrix}
=
\begin{bmatrix}
f_\theta(h(t),t) \\
-a(t)^\top \frac{\partial f_\theta}{\partial h} \\
-a(t)^\top \frac{\partial f_\theta}{\partial \theta}
\end{bmatrix}.
$$

The key tradeoff:

$$
\text{lower memory}
\quad \leftrightarrow \quad
\text{extra solves and possible numerical gradient error}.
$$

## Continuous Normalizing Flow

The paper also connects Neural ODEs to density modeling. If a variable $z(t)$ evolves by:

$$
\frac{dz(t)}{dt}
=
f_\theta(z(t),t),
$$

then its log-density changes according to the instantaneous change-of-variables formula:

$$
\frac{d \log p(z(t))}{dt}
=
-
\operatorname{Tr}
\left(
\frac{\partial f_\theta}{\partial z(t)}
\right).
$$

So:

$$
\log p(z(t_1))
=
\log p(z(t_0))
-
\int_{t_0}^{t_1}
\operatorname{Tr}
\left(
\frac{\partial f_\theta}{\partial z(t)}
\right)
dt.
$$

This defines a continuous normalizing flow. Unlike discrete normalizing flows, the transformation does not need a handcrafted invertible layer with easy Jacobian determinant. The ODE dynamics give invertibility under suitable conditions, and the log-density evolves through the trace term.

## Latent ODE

The paper also explores time-series modeling with latent dynamics. A latent initial state is inferred from observations:

$$
z(t_0) \sim q_\phi(z(t_0) \mid x_{1:n}),
$$

then evolved through:

$$
z(t_i)
=
\operatorname{ODESolve}
\left(
f_\theta, z(t_0), t_0, t_i
\right).
$$

Observations are decoded from the latent state:

$$
x_i \sim p_\psi(x_i \mid z(t_i)).
$$

This is useful for irregular time series because the model can query arbitrary times $t_i$ rather than fixed discrete steps.

## Evidence Reading

The paper presents Neural ODEs as a new architecture family, not as a claim that continuous depth always beats discrete depth.

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| Continuous-depth models can learn useful representations | supervised experiments and toy problems | ODE layers can replace some residual stacks | not always faster or more accurate |
| Adjoint method reduces memory | gradient method design and experiments | training can avoid storing all solver states | backward solve can be numerically fragile |
| Continuous normalizing flows are possible | density modeling examples | ODE dynamics can model invertible transformations | trace estimation and solver cost matter |
| Irregular time series can be modeled naturally | latent ODE experiments | arbitrary-time dynamics are useful | inference model quality is crucial |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task family | continuous-depth networks, density modeling, irregular time series |
| Architecture family | ODE-defined neural dynamics |
| Main comparison | residual networks and discrete normalizing flows |
| Core object | $\frac{dh}{dt}=f_\theta(h,t)$ |
| Main compute metric | number of function evaluations |
| Main memory trick | adjoint sensitivity method |
| Key hyperparameters | solver type, tolerance, integration interval, dynamics network |
| Not the claim | universal accuracy improvement over fixed-depth networks |

## Comparison to ResNet

| Property | ResNet | Neural ODE |
| --- | --- | --- |
| Depth | fixed integer layers | continuous integration interval |
| Update | $h_{k+1}=h_k+f(h_k)$ | $\frac{dh}{dt}=f(h,t)$ |
| Compute | fixed by layer count | chosen by solver and tolerance |
| Memory | activations stored or checkpointed | adjoint can reduce storage |
| Numerical issue | standard network stability | solver error, stiffness, tolerance |
| Interpretability | discrete blocks | continuous trajectory |

Neural ODE is not simply "a deeper ResNet." It changes who controls computation: the architecture designer or the numerical solver.

## Comparison to RNN

An RNN updates hidden states at discrete observations:

$$
h_i = \operatorname{RNNCell}(h_{i-1}, x_i).
$$

A latent ODE evolves hidden state between observations:

$$
h(t_i)
=
\operatorname{ODESolve}(f_\theta,h(t_{i-1}),t_{i-1},t_i).
$$

This helps when observations are irregular:

$$
t_{i+1}-t_i \text{ is not constant}.
$$

The model can separate:

- continuous latent evolution;
- observation updates;
- decoding at requested times.

## Solver Choice

The solver is part of the model behavior.

| Solver Property | Effect |
| --- | --- |
| fixed-step | predictable compute, less adaptive |
| adaptive-step | variable compute, tolerance-sensitive |
| high-order | fewer steps for smooth dynamics, more work per step |
| stiff-aware | handles difficult dynamics, may be expensive |
| loose tolerance | faster but less accurate |
| tight tolerance | more accurate but slower |

Two Neural ODE models with the same $f_\theta$ but different solvers can behave differently.

## Numerical Risks

### Stiffness

If dynamics require very small steps for stability, the solver may become slow:

$$
\operatorname{NFE} \uparrow.
$$

### Gradient Mismatch

The backward adjoint solve may not exactly reconstruct the forward trajectory, especially with adaptive solvers and numerical error.

### Hidden Compute

A model may look shallow in parameter count but expensive in function evaluations.

### Tolerance Overfitting

Changing solver tolerance at test time can change predictions. This means tolerance is part of the evaluation contract.

## Implementation Notes

### Track NFE

Always log:

$$
\operatorname{NFE}_{\text{forward}},
\qquad
\operatorname{NFE}_{\text{backward}}.
$$

Accuracy without NFE can be misleading.

### Report Solver and Tolerance

A reproducible Neural ODE result should specify:

- solver method;
- relative tolerance;
- absolute tolerance;
- integration interval;
- whether adjoint is used;
- maximum step or safety settings if relevant.

### Batch Behavior

Adaptive solvers can behave differently across batch elements. Batched solves may be constrained by the hardest example in the batch.

### Discrete Events

If the system has discontinuities, jumps, or event-driven changes, a plain smooth ODE dynamics model may be a poor fit without event handling.

## Molecular and Structural Modeling Reading

Neural ODEs are relevant to molecular modeling because molecules and proteins are often described by dynamics:

- molecular dynamics trajectories;
- conformational changes;
- diffusion-like generative paths;
- probability-flow ODEs;
- continuous normalizing flows;
- learned force or velocity fields.

But Neural ODE is not automatically a molecular dynamics simulator. A physical MD system has structure:

$$
\frac{dq}{dt}=v,
\qquad
\frac{dv}{dt}=M^{-1}F(q).
$$

Plain Neural ODE dynamics:

$$
\frac{dh}{dt}=f_\theta(h,t)
$$

do not guarantee conservation laws, equivariance, stability, or physical validity. For structure-based models, Neural ODE ideas are most useful when combined with:

- equivariant dynamics;
- energy-based constraints;
- graph or geometric representations;
- physically meaningful coordinates;
- careful numerical integration.

## Common Misreadings

### "Continuous depth is always better than discrete depth."

No. It is a different parameterization of computation. Discrete networks are often faster, simpler, and easier to optimize.

### "Adjoint backprop gives free memory savings."

It saves memory by recomputing through a backward solve. That can cost time and introduce numerical error.

### "The ODE solver is an implementation detail."

No. Solver, tolerance, and integration interval affect predictions, gradients, speed, and reproducibility.

### "Neural ODEs are automatically physically correct."

No. They are neural parameterizations of dynamics. Physical validity requires additional structure.

## Later-Paper Checklist

When reading later continuous-depth, flow, diffusion, or dynamics papers, ask:

- What is the state variable?
- What dynamics are being parameterized?
- Is time continuous for modeling reasons or mainly architectural style?
- What solver and tolerance are used?
- How many function evaluations are required?
- Are gradients computed by adjoint, checkpointing, direct backprop, or another method?
- Is the system stiff or numerically sensitive?
- Does the model need invariance, equivariance, conservation, or reversibility?
- Are wall-clock time and memory compared fairly to discrete baselines?
- Does changing solver tolerance change the reported result?

## Why It Matters

Neural ODE is important because it changed how architecture can be specified:

$$
\text{network} = \text{dynamics} + \text{solver}.
$$

It also connects several areas:

- residual networks as dynamical systems;
- memory-efficient training through adjoints;
- continuous normalizing flows;
- probability-flow ODEs in generative modeling;
- irregular time-series modeling;
- learned dynamics for scientific ML.

For this wiki, it is the anchor paper for continuous-depth architecture.

## Limitations

- Solver cost can dominate runtime.
- Adaptive computation makes benchmarking less straightforward.
- Adjoint gradients can be numerically unstable.
- Continuous-depth models do not automatically outperform fixed-depth models.
- Physical interpretability requires extra constraints.
- ODE solvers assume smooth dynamics; discontinuous or event-heavy systems may need different tools.

## Connections

- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/residual-network|Residual network]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/molecular-modeling/molecular-dynamics|Molecular dynamics]]
- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/index|Architecture papers]]
