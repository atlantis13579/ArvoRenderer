#include "Distributed/Protocol.h"
#include <limits>

namespace Arvo::Distributed
{
bool HasInstanceMotion(const RenderConfig& config)
{
    return std::any_of(config.InstanceMotions.begin(), config.InstanceMotions.end(),
                       [](const InstanceMotionOverride& item) { return item.Motion.Enabled != 0; });
}
namespace Protocol
{
namespace
{
constexpr size_t MaximumMotionOverrides = 4096;
constexpr size_t MotionFieldCount = 16; // Instance, enabled, then quaternion/translation for each key.
} // namespace
// Canonical order makes equivalent GUI/CLI edits share cache and retry identity.
// Scene indices refer to visible mesh placements in the instanced demo graph.
std::string EncodeInstanceMotions(std::vector<InstanceMotionOverride> motions)
{
    if (motions.size() > MaximumMotionOverrides)
        throw std::runtime_error("Too many instance motion overrides");
    std::sort(motions.begin(), motions.end(), [](const auto& a, const auto& b) { return a.Index < b.Index; });
    std::string result;
    uint32_t previous = 0;
    bool first = true;
    for (const auto& entry : motions)
    {
        if (!first && previous == entry.Index)
            throw std::runtime_error("Duplicate instance motion override");
        std::string error;
        if (!RayTracing::ValidateRigidMotion(entry.Motion, error))
            throw std::runtime_error(error);
        if (!first)
            result += ';';
        result += std::to_string(entry.Index) + ',' + std::to_string(entry.Motion.Enabled);
        for (const auto& pose : {entry.Motion.Transform.Start, entry.Motion.Transform.End})
            result += ',' + Number(pose.Rotation.X) + ',' + Number(pose.Rotation.Y) + ',' +
                      Number(pose.Rotation.Z) + ',' + Number(pose.Rotation.W) + ',' +
                      VectorText(pose.Translation);
        previous = entry.Index;
        first = false;
    }
    return result;
}
std::vector<InstanceMotionOverride> ParseInstanceMotions(const std::string& text)
{
    std::vector<InstanceMotionOverride> result;
    size_t begin = 0;
    while (begin < text.size())
    {
        if (result.size() == MaximumMotionOverrides)
            throw std::runtime_error("Too many instance motion overrides");
        const auto end = text.find(';', begin);
        const auto record = text.substr(begin, end == std::string::npos ? end : end - begin);
        std::map<std::string, std::string> fields;
        size_t cursor = 0;
        for (size_t field = 0; field < MotionFieldCount; ++field)
        {
            const auto separator = record.find(',', cursor);
            if ((field + 1 < MotionFieldCount) == (separator == std::string::npos))
                throw std::runtime_error("Incomplete instance motion record");
            fields[std::to_string(field)] =
                record.substr(cursor, separator == std::string::npos ? separator : separator - cursor);
            cursor = separator == std::string::npos ? record.size() : separator + 1;
        }
        InstanceMotionOverride item;
        item.Index = UInt(fields, "0", 0, std::numeric_limits<uint32_t>::max());
        item.Motion.Enabled = UInt(fields, "1", 0, 1);
        cursor = 2;
        const auto next = [&] { return Float(fields, std::to_string(cursor++)); };
        for (auto* pose : {&item.Motion.Transform.Start, &item.Motion.Transform.End})
        {
            pose->Rotation = {next(), next(), next(), next()};
            pose->Translation = {next(), next(), next()};
        }
        result.push_back(item);
        if (end == std::string::npos)
            break;
        begin = end + 1;
        if (begin == text.size())
            throw std::runtime_error("Empty instance motion record");
    }
    (void)EncodeInstanceMotions(result);
    return result;
}
void ApplyInstanceMotions(RayTracing::Scene& scene, const std::vector<InstanceMotionOverride>& motions)
{
    (void)EncodeInstanceMotions(motions);
    for (const auto& item : motions)
        if (item.Index >= scene.Instances.size())
            throw std::runtime_error("Instance motion index is out of range");
    auto candidate = scene;
    for (const auto& item : motions)
        candidate.Instances[item.Index].Motion = item.Motion;
    std::string error;
    if (!RayTracing::ValidateSceneMotionSupport(RayTracing::Algorithm::PathTracing, candidate, error))
        throw std::runtime_error(error);
    scene.Instances = std::move(candidate.Instances);
}
} // namespace Protocol
} // namespace Arvo::Distributed
