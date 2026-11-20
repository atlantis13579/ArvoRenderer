# Arvo

Arvo is an educational **distributed renderer** for light transport. Workers on
different platforms can render tiles of the same image in parallel, and Arvo
assembles them into the final frame. Offline rendering supports CPU, CUDA,
DirectCompute, DXR, and Metal. Tiled rendering scales to distributed workers,
render farms, and hybrid CPU/GPU jobs. Backends implement their own numerical
algorithms and share configuration and scene data to preserve sample identity
across tiles and retries. Stateful methods schedule complete photon passes,
light-path populations, or Markov chains. Arvo also explores physically based
shading and interactive approximations in scenes such as Cornell Box, Sponza,
and a Stanford Dragon study of glass, rough transmission, and subsurface
scattering.


<img width="1280" height="1265" alt="cornell-path-bvh-default-flat-1795066065269" src="https://github.com/user-attachments/assets/42e68491-e7a9-4f48-b466-2beea4121820" />


## Light transport

Arvo covers direct lighting, diffuse interreflection, specular transport, and caustics.
Photon and point-based reconstruction trade bias for efficiency. Progressive photon
methods reduce reconstruction bias as sampling increases.


<img width="2692" height="1840" alt="comparison" src="https://github.com/user-attachments/assets/95281db2-bf81-4c35-9f8a-e570aa228005" />


| Algorithm | Principle |
| --- | --- |
| Whitted-style ray tracing | Direct lighting and recursive specular reflection and refraction. |
| Distributed ray tracing | Stochastic sampling of glossy surfaces, finite lights, lens aperture, and shutter time. |
| Path tracing | Monte Carlo integration along camera paths. |
| Bidirectional path tracing (BDPT) | Connects camera and light subpaths with multiple importance sampling. |
| Metropolis light transport (MLT) | Mutates samples in primary sample space to favor contributing light paths. |
| Vertex connection and merging (VCM) | Weights bidirectional connections and photon-style vertex merging together. |
| Photon mapping | Estimates indirect lighting and caustics from the density of photons traced from lights. |
| Progressive photon mapping (PPM) | Accumulates flux across photon passes as the reconstruction radius shrinks. |
| Point-based global illumination | Stores and gathers approximate illumination from surface samples in a spatial hierarchy. |


<img width="1136" height="2640" alt="detail-comparison" src="https://github.com/user-attachments/assets/f7c2879c-40f4-4976-ac37-f94f263f0a89" />


## Sampling and convergence

- Importance sampling for scattering and lights, next-event estimation (NEE), and multiple
  importance sampling (MIS) with balance and power heuristics.
- Independent random, stratified, Sobol quasi-Monte Carlo, and procedural blue-noise
  sampling.
- Russian roulette for stochastic path termination.
- Variance estimation, adaptive sampling, and denoiser-aware sample allocation.

Sampling, transport, and reconstruction are separate choices within estimator constraints.
Independent path samples, correlated Markov chains, and progressive photon passes need
different convergence models.

## Materials, media and camera effects

Surface scattering covers diffuse materials, conductors, and dielectrics, with GGX
microfacet reflection and transmission, Fresnel response, and importance sampling of
visible microfacet normals. Transport supports RGB and spectral modes, with spectral
transport resolving wavelength-dependent light and material response.


<img width="1280" height="720" alt="dragon-path-bvh-default-flat-1794290460079" src="https://github.com/user-attachments/assets/9d632b64-159e-4ec4-bd89-3a449f0f0622" />


- Homogeneous participating media require path tracing and support absorption, multiple
  scattering, Beer-Lambert transmittance, and the Henyey-Greenstein phase function.
- Subsurface random walks require static, closed dielectric solids and full geometric
  shadow visibility. Camera motion is supported.
- Whitted tracing supports diffuse and ideal specular surfaces. Point-based GI uses a
  view-independent diffuse cache and supports subsurface walks in dielectrics.
- Thin-lens depth of field samples a finite aperture. Shutter-time sampling produces
  camera and rigid-object motion blur.
- Micropolygon dicing supports fine geometry and procedural displacement. Authored height
  maps and measured material spectra are planned.

## Geometry and acceleration

Spatial partitioning uses bounding volume hierarchies (BVHs), spatial-split BVHs, k-d
trees, and bounding interval hierarchies (BIHs). Instancing shares geometry through
bottom- and top-level acceleration structures and transforms rays into object space.
Motion bounds span the shutter interval.

## Reconstruction

Spatial a-trous filtering uses radiance variance and optional primary-surface guides. On
RGB diffuse surfaces, it weights radiance and multisampled albedo together to preserve
texture contrast while smoothing lighting noise. Temporal reconstruction reprojects
history, rejects incompatible observations, and clips outliers. Adaptive reconstruction
uses uncertainty estimates to allocate samples. AI denoising is planned.

## Interactive preview

DXR previews trace short reflection and refraction paths at native viewport resolution
with simplified lighting. DX11 raster previews use physical shading, shadow maps, and
approximate environment lighting. Previews prioritize responsiveness. Offline rendering
models light transport more fully.
