# Arvo

Arvo is an educational renderer exploring offline light transport, physically
based shading and interactive approximations. Its scenes include Cornell Box,
Sponza and a Stanford Dragon study of glass, rough transmission and subsurface
scattering.

## Light transport

| Algorithm | Principle |
| --- | --- |
| Whitted-style ray tracing | Direct illumination with recursive specular reflection and refraction. |
| Distributed ray tracing | Stochastic sampling of glossy surfaces, finite lights, lens aperture and shutter time. |
| Path tracing | Monte Carlo integration of light transport along camera paths. |
| Bidirectional path tracing (BDPT) | Connections between camera and light subpaths, combined with multiple importance sampling. |
| Metropolis light transport (MLT) | Mutations in primary sample space concentrate exploration around contributing light paths. |
| Vertex connection and merging (VCM) | Combines bidirectional connections with photon-style vertex merging under a common weighting scheme. |
| Photon mapping | Light-emitted paths populate a density estimate for indirect illumination and caustics. |
| Progressive photon mapping (PPM) | Repeated photon passes accumulate flux while progressively shrinking the reconstruction radius. |
| Point-based global illumination | Surface samples store and gather approximate illumination through a spatial hierarchy. |

These methods cover direct lighting, diffuse interreflection, specular transport
and caustics. Photon and point-based reconstruction trade bias for efficiency;
progressive photon methods reduce reconstruction bias as sampling increases.

## Sampling and convergence

- Importance sampling of scattering and light distributions.
- Next event estimation (NEE) and multiple importance sampling (MIS), including
  balance and power heuristics.
- Independent random, stratified, Sobol quasi-Monte Carlo and procedural
  blue-noise sampling.
- Russian roulette for stochastic path termination.
- Adaptive sampling, variance estimation and denoiser-aware sample allocation.

Sampling, transport and reconstruction are separate choices, subject to the
estimator's compatibility constraints. Correlated Markov chains, progressive
photon passes and independent path samples require different convergence models.

## Materials, media and camera effects

Physically based surface scattering includes diffuse materials, conductors and
dielectrics, GGX microfacet reflection and transmission, Fresnel response and
importance sampling of visible microfacet normals.

RGB and spectral transport represent wavelength-dependent light and material
response. Subsurface scattering uses interior random walks. Homogeneous
participating media support absorption, multiple scattering, Beer-Lambert
transmittance and the Henyey-Greenstein phase function.

Homogeneous participating media currently require path tracing. Subsurface walks
require static, closed dielectric solids and full geometric shadow visibility;
camera motion is supported. Whitted tracing supports diffuse and ideal specular
surfaces. Point-based GI uses a view-independent diffuse cache, with subsurface
walks for dielectric interiors.

Thin-lens depth of field samples a finite aperture. Shutter-time sampling produces
camera and rigid-object motion blur. Micropolygon dicing supports procedural
displacement and fine geometric detail.
Displacement is procedural; authored height maps and measured material spectra
remain planned capabilities.

## Geometry and acceleration

Bounding volume hierarchies (BVH), spatial-split BVHs, k-d trees and bounding
interval hierarchies (BIH) provide different spatial partitioning strategies.
Instancing shares object geometry through bottom-level and top-level acceleration
structures, with rays transformed into object space. Motion bounds cover geometry
throughout the shutter interval.

## Reconstruction

Spatial a-trous filtering uses radiance variance and geometric guides. Temporal
reconstruction reprojects history, rejects incompatible observations and clips
outliers. Adaptive reconstruction couples uncertainty estimates to sampling.
AI-based denoising remains a planned capability.

## Execution and preview

Offline rendering supports CPU, CUDA, DirectCompute, DXR and Metal. Tile-based
execution extends to distributed workers, render farms and hybrid CPU/GPU jobs.
Each backend owns its numerical algorithms. Shared configuration and scene data
preserve sample identity across tiles and retries. Stateful methods schedule
complete photon passes, light-path populations or Markov chains.

DXR supports both offline transport and interactive ray-traced previews. Preview
uses native viewport resolution with simplified lighting and short reflection
and refraction paths. DX11 provides a raster preview with physical shading,
shadow maps and approximate environment lighting. Preview favors responsiveness;
offline rendering provides the more complete light transport model.

[Asset credits](Contents/ATTRIBUTION.md)
