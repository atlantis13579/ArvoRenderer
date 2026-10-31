# Arvo

Arvo is an independent C++17 offline renderer with an interactive preview and
HTTP render workers. Its only scenes are Cornell Box and Sponza. All source,
assets, shaders and build definitions are contained in this repository.

## Algorithms and backends

Ray Tracing computes direct light with recursive mirror reflection. Path Tracing
uses cosine-weighted diffuse continuation, stochastic mirror selection, explicit
light sampling and emissive surfaces. CPU, CUDA, Compute Shader and Metal use the
same material/light/camera contract, and DXR implements both algorithms for
preview. DX11 is a raster preview and does not simulate indirect illumination.

Sampling is indexed by full-frame pixel coordinates. A tile carries the full
image dimensions and its origin, so changing worker count, tile size or task
ordering preserves the image. The client combines RGBA tiles and saves one PNG
only after every tile completes. Cancellation and exhausted retries do not save
a partial render as a completed image.

The Windows frontend offers DX11 and DXR; macOS uses Metal to present the
preview. CPU is portable, CUDA requires an NVIDIA GPU, Compute Shader uses
Direct3D on Windows, and Metal requires macOS. Unavailable backends return an
error; they do not silently render on the CPU.

Workers serve installed Cornell/Sponza assets from their own copy of the scene
data, and the protocol verifies scene identity before combining tiles, so every
worker must run the same build and assets as the client. The protocol is HTTP;
there is no built-in TLS.

Asset attribution is recorded in `Contents/ATTRIBUTION.md`.
