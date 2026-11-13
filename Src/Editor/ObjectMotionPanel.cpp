#include "Editor/Imgui.h"
#include "Editor/RenderPanel.h"
#include <algorithm>
#include <cmath>

namespace Arvo::Editor
{
void RenderPanel::DrawObjectMotion(EditorCore& editor)
{
    if (imguiCollapse("Object motion", "", ObjectMotionOpen_))
        ObjectMotionOpen_ = !ObjectMotionOpen_;
    if (!ObjectMotionOpen_)
        return;
    const auto& scene = editor.Scene();
    std::vector<const SceneNode*> nodes;
    std::vector<const char*> names;
    for (const auto& node : scene.Nodes)
        if (node.Visible && node.Mesh && !node.Mesh->Indices.empty())
        {
            nodes.push_back(&node);
            names.push_back(node.Name.c_str());
        }
    if (nodes.empty())
    {
        imguiValue("No visible mesh objects");
        return;
    }
    MotionNode_ = std::clamp(MotionNode_, 0, int(nodes.size()) - 1);
    imguiCombo("Moving object", &MotionNode_, names.data(), int(names.size()), !editor.IsDemoLoading());
    const auto& node = *nodes[MotionNode_];
    auto motion = node.Motion;
    const bool motionSupported = RayTracing::SupportsMotionBlur(static_cast<RayTracing::Algorithm>(Algorithm_));
    std::string reason;
    const bool movable = CanAnimateNode(scene, node.Id, reason);
    const bool editable = motionSupported && movable && !editor.IsDemoLoading();
    bool changed = false;
    constexpr float MaximumCoordinate = 1000000, MinimumRange = 1, DefaultShiftFraction = 0.05f;
    constexpr float SliderResolution = 1000, MaximumYaw = 180, AngularStep = 0.1f;
    constexpr float DegreesToHalfRadians = 3.14159265358979323846f / 360;
    float range = MinimumRange;
    for (const auto& vertex : node.Mesh->Vertices)
        range = std::max(
            {range, std::abs(vertex.Position.X), std::abs(vertex.Position.Y), std::abs(vertex.Position.Z)});
    range = std::min(range, MaximumCoordinate);
    if (imguiCheck("Rigid object motion", motion.Enabled != 0, editable))
    {
        motion.Enabled = !motion.Enabled;
        if (motion.Enabled && motion.Transform.End.Translation.X == 0 &&
            motion.Transform.End.Translation.Y == 0 && motion.Transform.End.Translation.Z == 0 &&
            motion.Transform.End.Rotation.X == 0 && motion.Transform.End.Rotation.Y == 0 &&
            motion.Transform.End.Rotation.Z == 0)
            motion.Transform.End.Translation.X = range * DefaultShiftFraction;
        changed = true;
    }
    if (!motionSupported)
        imguiValue("Object motion: Path or Distributed Ray");
    if (!movable)
        imguiValue(reason.c_str());
    if (motion.Enabled)
    {
        const float step = range / SliderResolution;
        changed |= imguiSlider("Object end shift X", &motion.Transform.End.Translation.X, -range, range, step,
                               editable);
        changed |= imguiSlider("Object end shift Y", &motion.Transform.End.Translation.Y, -range, range, step,
                               editable);
        changed |= imguiSlider("Object end shift Z", &motion.Transform.End.Translation.Z, -range, range, step,
                               editable);
        float yaw = std::atan2(motion.Transform.End.Rotation.Y, motion.Transform.End.Rotation.W) /
                    DegreesToHalfRadians;
        if (imguiSlider("Object end yaw", &yaw, -MaximumYaw, MaximumYaw, AngularStep, editable))
        {
            motion.Transform.End.Rotation = {0, std::sin(yaw * DegreesToHalfRadians), 0,
                                             std::cos(yaw * DegreesToHalfRadians)};
            changed = true;
        }
        imguiValue("Local units; rotation about object origin");
        imguiValue("Uses the camera shutter; instance geometry required");
        imguiValue("DXR: exposure integration; DX11: midpoint pose");
    }
    if (changed)
    {
        try
        {
            editor.SetNodeMotion(node.Id, motion);
            if (motion.Enabled)
                Geometry_ = int(RayTracing::GeometryMode::Instanced);
            Error_.clear();
        }
        catch (const std::exception& error)
        {
            Error_ = error.what();
        }
    }
    if (imguiButton("Disable all object motion", HasObjectMotion(editor.Scene()) && !editor.IsDemoLoading()))
        editor.DisableObjectMotion();
}
} // namespace Arvo::Editor
