#pragma once

#include "Editor/Resources.hpp"

#include <filesystem>
#include <string_view>
#include <vector>

namespace Arvo::Editor
{

struct DemoDescriptor
{
    DemoId Id = DemoId::CornellBox;
    std::string_view Key;
    std::string_view Name;
    std::string_view Description;
};

[[nodiscard]] std::string_view DemoKey(DemoId demo);
[[nodiscard]] bool ParseDemo(std::string_view key, DemoId& demo);

class DemoCatalog
{
public:
    explicit DemoCatalog(std::filesystem::path contentRoot);

    [[nodiscard]] static const std::vector<DemoDescriptor>& Descriptors();
    [[nodiscard]] SceneSnapshot Load(DemoId demo) const;

private:
    [[nodiscard]] SceneSnapshot LoadDragonDemo(const std::filesystem::path& relativePath) const;
    [[nodiscard]] SceneSnapshot LoadModelDemo(
        DemoId demo,
        const std::filesystem::path& relativePath) const;

    std::filesystem::path ContentRoot_;
};

} // namespace Arvo::Editor
