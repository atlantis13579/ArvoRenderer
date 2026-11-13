#pragma once

#include "Editor/Editor.hpp"
#include "Renderer/RayTracing.h"

namespace Arvo::Editor
{

std::shared_ptr<const RayTracing::Scene> BuildOfflineScene(const SceneSnapshot& scene,
    RayTracing::GeometryMode geometry = RayTracing::GeometryMode::Flat);
RayTracing::Camera BuildOfflineCamera(const SceneSnapshot& scene);
// Restore authored node IDs/materials around a native diced mesh for raster upload.
std::shared_ptr<const SceneSnapshot> BuildMicropolygonPreviewScene(const SceneSnapshot& scene,
    const RayTracing::MicropolygonResult& diced, const CancellationToken& cancellation);
[[nodiscard]] std::shared_ptr<IRenderAdapter> CreateRayRenderAdapter(
    BackendDescriptor descriptor, std::shared_ptr<RayTracing::IRayTracingRHI> rhi);
// Production DXR preview uses bounded approximations; the exact adapter above
// remains useful for offline scene conversion and transport validation.
[[nodiscard]] std::shared_ptr<IRenderAdapter> CreateDxrPreviewAdapter(
    BackendDescriptor descriptor, std::shared_ptr<RayTracing::IRayTracingRHI> rhi);

void RegisterModuleAdapters(EditorCore& editor);
[[nodiscard]] PickHandler CreatePickHandler();

} // namespace Arvo::Editor
