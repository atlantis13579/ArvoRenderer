#pragma once

#include "Editor/Editor.hpp"
#include "Renderer/RayTracing.h"

namespace Arvo::Editor
{

std::shared_ptr<const RayTracing::Scene> BuildOfflineScene(const SceneSnapshot& scene);
RayTracing::Camera BuildOfflineCamera(const SceneSnapshot& scene);

void RegisterModuleAdapters(EditorCore& editor);
[[nodiscard]] PickHandler CreatePickHandler();

} // namespace Arvo::Editor
