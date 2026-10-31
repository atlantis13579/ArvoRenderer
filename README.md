# Arvo

Arvo is an independent C++17 offline renderer with an interactive preview and
HTTP render workers. Its only scenes are Cornell Box and Sponza. All source,
assets, shaders and build definitions are contained in this repository.

## Build

Windows requires Visual Studio 2022 with the C++ desktop workload, Windows SDK
and CMake. Shader bytecode is embedded; the CUDA Toolkit and DXC are needed only
when regenerating shaders, not when building or running the editor.

```powershell
.\Build\GenerateVS2022.bat
cmake --build Build/VS2022 --config Release --parallel
ctest --test-dir Build/VS2022 -C Release --output-on-failure
.\Binaries\Release\ArvoEditor.exe
```

Start two local GPU worker processes before clicking Render:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File Build/StartLocalWorkers.ps1 -Backend compute
```

Their endpoints default to `http://127.0.0.1:7831` and `http://127.0.0.1:7832`.
`-Backend cuda` selects CUDA. Workers run with hidden console windows; PID files
and logs are under `Intermediate/Workers`. Run `Build/StopLocalWorkers.ps1` to
stop workers started by the launcher.

To render without the editor:

```powershell
.\Binaries\Release\ArvoRender.exe --servers http://127.0.0.1:7831,http://127.0.0.1:7832 --backend compute --scene cornell --algorithm path --width 3840 --height 2160 --samples 16 --output Binaries/Renders/cornell.png
```

The same worker/CLI executables run on macOS without the `.exe` suffix. Start
`ArvoWorker --port 7831 --backend metal` and another instance on port 7832.

On macOS with Xcode and CMake:

```sh
sh Build/GenerateXcode.sh
cmake --build Build/Xcode --config Release --parallel
ctest --test-dir Build/Xcode -C Release --output-on-failure
Binaries/Release/ArvoEditor
```

`Src` contains source code, `Contents` contains the two scenes, `Build` contains
build definitions and generated IDE projects, `Intermediate` contains object
files, static libraries, symbols and validation output, and `Binaries` contains
executables and saved renders. Generated directories are ignored by Git.

## Preview and rendering

The Windows frontend offers DX11 and DXR. macOS uses Metal to present the
preview. WASD moves the camera, right drag rotates it, and left click picks scene
geometry. The viewport uses the same 16:9 framing as the output presets.

The ImGui panel selects the preview, scene, Ray Tracing or Path Tracing, preview
sample count, path depth, worker backend, output resolution, offline sample count
and tile size. Edit HTTP worker URLs and an optional access token directly in the
panel. Click a text field to edit it; Ctrl+A clears it and Enter finishes editing.
The output defaults to 3840 x 2160 PNG. Render PNG snapshots the camera/settings
and runs asynchronously; navigation remains responsive. Output filenames include
a timestamp and are written under `Binaries/Renders` by default.

Each server row reports completed tiles, status, worker-process CPU utilization
and device-wide GPU utilization. GPU utilization uses NVML when available; an
unavailable reading is shown as N/A. Two workers on the same physical GPU report
the same device-wide utilization, not independent per-process GPU percentages.

## Algorithms and workers

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

Workers serve installed Cornell/Sponza assets from their own `Contents` directory.
Deploy the same build and assets to each worker; the protocol verifies scene
identity before combining tiles. Start each worker with the backend selected in
the frontend, or select Per server (auto) to use each server's configured backend.
CPU is portable, CUDA requires an NVIDIA GPU, Compute Shader uses
Direct3D on Windows, and Metal requires macOS. Unavailable backends return an
error; they do not silently render on the CPU.

Deployment can preserve `Binaries/Release` with `Contents` at the repository root,
or place the executables beside `Contents`. Runtime resource discovery supports
both layouts. `--content-root` explicitly overrides discovery on the worker and
CLI. For example:

```sh
ArvoWorker --bind 0.0.0.0 --port 7831 --backend compute --content-root /deployment/Contents
```

Workers default to loopback. For a cloud deployment, configure a reachable bind
address and token, and place the HTTP endpoint behind your network's authenticated
TLS proxy or private tunnel. The protocol is HTTP; there is no built-in TLS.

## Shader maintenance

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File Build/GenerateRayTracingShaders.ps1
```

Shader source and embedded DXIL/PTX headers are versioned together. Object files,
generated IDE files, render outputs, logs and local tokens must not be committed.
Asset attribution is recorded in `Contents/ATTRIBUTION.md`.

## Validation

CTest covers integrator parity, tile determinism, actual scene/DXR preview parity,
atomic PNG publication and the native Windows preview. Distributed tests start
two isolated workers, validate both algorithms and both scenes, exercise failed
endpoints, authentication, invalid requests and cancellation, then decode a 4K PNG:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File Build/TestLocalWorkers.ps1 -Backend compute
powershell -NoProfile -ExecutionPolicy Bypass -File Build/TestLocalWorkers.ps1 -Backend cuda
powershell -NoProfile -ExecutionPolicy Bypass -File Build/TestLocalWorkers.ps1 -Backend cpu -Skip4K
```

The distributed test uses Python 3's standard library and writes all evidence to
`Intermediate/Validation`. The client verifies every worker's scene hash against
the preview's assets. Output PNGs are published atomically so a failed save cannot
destroy the previous complete image.

Windows validation and known platform limits are recorded in `Build/Validation.md`.
