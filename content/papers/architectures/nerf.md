---
title: NeRF
aliases:
  - papers/nerf
  - papers/neural-radiance-fields
  - papers/representing-scenes-as-neural-radiance-fields
tags:
  - papers
  - architectures
  - neural-fields
  - 3d-representation
---

# NeRF

> The paper represents a 3D scene as a continuous neural radiance field and renders images by differentiable volume rendering along camera rays.

## Metadata

| Field | Value |
| --- | --- |
| Paper | NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis |
| Authors | Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng |
| Year | 2020 |
| Venue | ECCV 2020 |
| arXiv | [2003.08934](https://arxiv.org/abs/2003.08934) |
| Project | [NeRF project page](https://www.matthewtancik.com/nerf) |
| Code | [bmild/nerf](https://github.com/bmild/nerf) |
| Status | full note started |

## One-Line Takeaway

NeRF represents a scene with a neural function:

$$
F_\theta:
(\mathbf{x},\mathbf{d})
\mapsto
(\sigma,\mathbf{c}),
$$

where $\mathbf{x}\in\mathbb{R}^3$ is a spatial coordinate, $\mathbf{d}$ is a viewing direction, $\sigma$ is volume density, and $\mathbf{c}\in\mathbb{R}^3$ is view-dependent color.

Images are rendered by sampling points along camera rays and integrating color/density with differentiable volume rendering.

## Question

Given multiple images of a scene with known camera poses:

$$
\{(I_i, \Pi_i)\}_{i=1}^{N},
$$

can a model synthesize a new view from a novel camera pose?

The architecture question is:

$$
\text{What representation should store the 3D scene?}
$$

Instead of explicit meshes, voxels, or point clouds, NeRF uses an implicit continuous field:

$$
\mathbf{x},\mathbf{d}
\rightarrow
\sigma,\mathbf{c}.
$$

## Main Claim

A fully connected neural network can represent a continuous volumetric scene function that supports photorealistic novel view synthesis when optimized through differentiable volume rendering.

The compact architecture claim:

$$
\text{coordinate MLP}
+
\text{positional encoding}
+
\text{volume rendering}
\Rightarrow
\text{continuous neural scene representation}.
$$

This is not a classifier or a conventional image generator. It is a scene-specific representation trained to reproduce observed views and render unseen views.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input data | posed RGB images of a scene |
| Query input | 3D point $\mathbf{x}$ and viewing direction $\mathbf{d}$ |
| Neural function | MLP radiance field $F_\theta$ |
| Output | density $\sigma$ and color $\mathbf{c}$ |
| Rendering | differentiable volume rendering along rays |
| Supervision | pixel reconstruction from known camera poses |
| Main task | novel view synthesis |
| Representation type | continuous implicit neural field |
| Main bottleneck | many MLP evaluations per rendered image |

NeRF is an architecture for representing a scene, not a generic object detector or image classifier.

## Radiance Field

The neural field maps:

$$
F_\theta(\mathbf{x},\mathbf{d})
=
(\sigma,\mathbf{c}).
$$

Here:

| Symbol | Meaning |
| --- | --- |
| $\mathbf{x}=(x,y,z)$ | spatial position in scene coordinates |
| $\mathbf{d}$ | viewing direction |
| $\sigma$ | volume density or opacity-like quantity |
| $\mathbf{c}=(r,g,b)$ | emitted color for that point and view direction |

Density depends mainly on position:

$$
\sigma=\sigma(\mathbf{x}),
$$

while color can depend on both position and direction:

$$
\mathbf{c}=\mathbf{c}(\mathbf{x},\mathbf{d}).
$$

This allows view-dependent appearance such as specular effects.

## Ray Parameterization

A camera ray is:

$$
\mathbf{r}(t)
=
\mathbf{o}
+
t\mathbf{d},
$$

where $\mathbf{o}$ is the camera origin and $\mathbf{d}$ is the ray direction.

The rendered color for a ray is:

$$
C(\mathbf{r})
=
\int_{t_n}^{t_f}
T(t)
\sigma(\mathbf{r}(t))
\mathbf{c}(\mathbf{r}(t),\mathbf{d})
dt.
$$

The transmittance is:

$$
T(t)
=
\exp
\left(
-
\int_{t_n}^{t}
\sigma(\mathbf{r}(s))ds
\right).
$$

Interpretation:

- high density contributes opacity;
- color contributes where the ray terminates or passes through density;
- transmittance downweights points behind dense regions.

## Discrete Rendering Approximation

In practice, sample points along the ray:

$$
t_1,\ldots,t_N.
$$

Let:

$$
\delta_i=t_{i+1}-t_i,
$$

and:

$$
\alpha_i
=
1-\exp(-\sigma_i\delta_i).
$$

The accumulated transmittance before sample $i$ is:

$$
T_i
=
\prod_{j<i}
(1-\alpha_j).
$$

The rendered color is approximated by:

$$
\hat{C}(\mathbf{r})
=
\sum_{i=1}^{N}
T_i\alpha_i\mathbf{c}_i.
$$

This equation is the bridge between the neural field and image pixels.

## Training Objective

For a set of camera rays $\mathcal{R}$ with ground-truth pixel colors $C(\mathbf{r})$, NeRF minimizes photometric reconstruction:

$$
\mathcal{L}
=
\sum_{\mathbf{r}\in\mathcal{R}}
\left\|
\hat{C}(\mathbf{r})
-
C(\mathbf{r})
\right\|_2^2.
$$

The model learns geometry and appearance indirectly from images and camera poses.

No explicit 3D mesh supervision is required:

$$
\text{posed images}
\rightarrow
\text{rendering loss}
\rightarrow
\text{scene field}.
$$

## Positional Encoding

A plain MLP can struggle to represent high-frequency detail. NeRF applies positional encoding to coordinates:

$$
\gamma(p)
=
\left(
\sin(2^0\pi p),
\cos(2^0\pi p),
\ldots,
\sin(2^{L-1}\pi p),
\cos(2^{L-1}\pi p)
\right).
$$

For a 3D coordinate:

$$
\gamma(\mathbf{x})
=
[\gamma(x),\gamma(y),\gamma(z)].
$$

This gives the MLP access to multiple spatial frequencies:

$$
\mathbf{x}
\rightarrow
\gamma(\mathbf{x})
\rightarrow
F_\theta.
$$

The architecture lesson is that coordinate networks often need input feature maps that expose high-frequency variation.

## Coarse And Fine Sampling

NeRF uses hierarchical sampling:

| Stage | Role |
| --- | --- |
| coarse network | sample broadly along the ray |
| fine network | resample more around likely occupied regions |

The coarse pass estimates where density lies. The fine pass allocates more samples to important ray intervals.

This is an efficiency and quality mechanism:

$$
\text{spend samples where the scene has density}.
$$

## Relation To PointNet And 3D Structure

[[papers/architectures/pointnet|PointNet]] consumes a finite unordered point set:

$$
X=\{x_1,\ldots,x_n\}.
$$

NeRF represents a continuous field:

$$
F_\theta(\mathbf{x},\mathbf{d})
$$

that can be queried at arbitrary coordinates.

| Axis | PointNet | NeRF |
| --- | --- | --- |
| object | point set | continuous scene field |
| input unit | points | coordinate and direction queries |
| output | class or per-point labels | density and view-dependent color |
| supervision | labels or point annotations | posed images |
| task | classification/segmentation | novel view synthesis |

For molecular or protein structures, NeRF is not directly a chemistry model. But the broader idea of coordinate-to-field representation is relevant to implicit surfaces, density fields, and differentiable rendering-style losses.

## Relation To Generative Models

NeRF is often used inside generative systems, but the original paper is not primarily a distribution model:

$$
\theta^\*
=
\arg\min_\theta
\mathcal{L}_{\text{render}}(\theta;\text{one scene}).
$$

It is usually optimized per scene from posed images.

This differs from models like [[papers/architectures/latent-diffusion-models|Latent Diffusion Models]], which learn a generative distribution over many images.

The reading boundary:

| Claim | NeRF Original Paper? |
| --- | --- |
| continuous scene representation | yes |
| differentiable novel view synthesis | yes |
| distribution over scenes | no, not the core original claim |
| text-to-3D generation | no, later systems |

## Why It Belongs In Architecture Papers

NeRF is a canonical architecture paper because it reframes scene representation:

$$
\text{explicit geometry}
\rightarrow
\text{implicit neural field}.
$$

It also establishes a reusable pattern:

| Component | Function |
| --- | --- |
| coordinate MLP | stores continuous scene properties |
| positional encoding | enables high-frequency detail |
| ray sampling | queries the field along camera rays |
| volume rendering | maps field values to pixels |
| photometric loss | trains from posed images |

This pattern appears in many later neural field, 3D reconstruction, view synthesis, avatar, robotics, and 3D-aware generation papers.

## Evidence Pattern

The paper supports the architecture with:

| Evidence | What It Supports |
| --- | --- |
| novel view synthesis benchmarks | rendered views match held-out camera views |
| comparisons to prior neural rendering | continuous radiance field improves fidelity |
| ablations on positional encoding | high-frequency coordinate encoding matters |
| coarse/fine sampling ablations | hierarchical sampling improves quality |
| qualitative video results | view consistency and photorealism are visible |

The core evidence is held-out view reconstruction, not generic image classification accuracy.

## Practical Reading Checks

| Question | Why |
| --- | --- |
| Are camera poses known or estimated? | pose errors can dominate quality |
| Is the model trained per scene or generalized across scenes? | changes the claim |
| How many samples per ray are used? | controls speed and quality |
| Is positional encoding used? | affects high-frequency detail |
| Are views sparse or dense? | sparse-view setting is harder |
| Are dynamic objects present? | static scene assumption may break |
| What metric is reported? | PSNR/SSIM/LPIPS measure different properties |

## Limits

- Original NeRF is slow because rendering requires many MLP evaluations.
- It assumes known camera poses.
- It is primarily a static scene representation.
- Sparse views can be underconstrained.
- It can produce artifacts under pose errors, reflections, transparency, or dynamic scenes.
- It does not directly provide editable semantic structure.
- It is not inherently a generative model over scene distributions.

The concise limitation:

$$
\text{continuous neural scene representation}
\neq
\text{fast, semantic, editable, general 3D understanding}.
$$

## What To Remember

- NeRF maps 5D coordinate-plus-view direction queries to density and color.
- Images are produced by differentiable volume rendering along camera rays.
- Training uses posed images and pixel reconstruction loss.
- Positional encoding is crucial for high-frequency detail.
- NeRF is an implicit neural field, not a point cloud or mesh.
- The paper created a major architecture pattern for neural scene representations.

## Links

- [[concepts/modalities/3d-structure|3D structure]]
- [[concepts/math/geometry|Geometry]]
- [[concepts/architectures/mlp|MLP]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[papers/architectures/pointnet|PointNet]]
- [[papers/architectures/latent-diffusion-models|Latent Diffusion Models]]
