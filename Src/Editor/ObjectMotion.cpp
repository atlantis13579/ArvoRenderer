#include "Editor/Editor.hpp"
#include <algorithm>
#include <stdexcept>

namespace Arvo::Editor
{
bool HasObjectMotion(const SceneSnapshot& scene)
{
    return std::any_of(scene.Nodes.begin(), scene.Nodes.end(), [](const SceneNode& node) {
        return node.Visible && node.Mesh && node.Motion.Enabled != 0;
    });
}
bool CanAnimateNode(const SceneSnapshot& scene, SceneNodeId id, std::string& error)
{
    const auto found = std::find_if(scene.Nodes.begin(), scene.Nodes.end(),
                                    [id](const SceneNode& node) { return node.Id == id; });
    if (found == scene.Nodes.end() || !found->Visible || !found->Mesh || found->Mesh->Indices.empty())
    {
        error = "Select a visible mesh for object motion";
        return false;
    }
    // Cornell's light mesh is paired with an explicit rectangle sampler. Their
    // trajectories must be authored together before either may be animated.
    if (scene.Demo == DemoId::CornellBox && !scene.Lights.empty())
        for (const auto& section : found->Mesh->Sections)
            if (section.MaterialName == "light" && section.IndexCount != 0)
            {
                error = "This emitter needs a synchronized light trajectory";
                return false;
            }
    return true;
}
void EditorCore::SetNodeMotion(SceneNodeId id, const RayTracing::MotionData::RigidMotion& motion)
{
    if (ActiveDemoLoad_)
        throw std::runtime_error("Wait for scene loading before editing object motion");
    std::string error;
    if (!RayTracing::ValidateRigidMotion(motion, error))
        throw std::invalid_argument(error);
    if (motion.Enabled && !CanAnimateNode(Scene_, id, error))
        throw std::invalid_argument(error);
    if (motion.Enabled && !RayTracing::SupportsMotionBlur(Settings_.Integrator))
        throw std::invalid_argument("Object motion requires Path or Distributed Ray Tracing");
    const auto found = std::find_if(Scene_.Nodes.begin(), Scene_.Nodes.end(),
                                    [id](const SceneNode& node) { return node.Id == id; });
    if (found == Scene_.Nodes.end())
        throw std::invalid_argument("Unknown scene object");
    found->Motion = motion;
    if (motion.Enabled)
    {
        Settings_.Geometry = RayTracing::GeometryMode::Instanced;
        Settings_.TemporalPreview = false;
        Settings_.Reconstruction.GeometryGuidance = false;
        if (Scene_.SceneCamera.Motion == RayTracing::MotionData::Exposure{})
        {
            Scene_.SceneCamera.Motion.Interval = {0, 1};
            ++Scene_.CameraRevision;
        }
    }
    ++Scene_.Revision;
    InvalidateAndRender();
}
void EditorCore::DisableObjectMotion()
{
    bool changed = false;
    for (auto& node : Scene_.Nodes)
    {
        changed |= node.Motion.Enabled != 0;
        node.Motion.Enabled = 0;
    }
    if (!changed)
        return;
    ++Scene_.Revision;
    InvalidateAndRender();
}
} // namespace Arvo::Editor
