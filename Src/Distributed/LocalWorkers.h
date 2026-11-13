#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace Arvo::Distributed
{
// Own only the child processes started by this editor. Existing compatible
// workers are borrowed; remote URLs always retain their normal client behavior.
class LocalWorkers
{
public:
    LocalWorkers(std::filesystem::path contentRoot, std::string token,
                 std::array<uint16_t, 2> ports = {7831, 7832}, std::filesystem::path executable = {});
    ~LocalWorkers();
    LocalWorkers(const LocalWorkers&) = delete;
    LocalWorkers& operator=(const LocalWorkers&) = delete;
    void EnsureReady(const std::vector<std::string>& servers, const std::atomic<bool>* cancelled = nullptr);
    [[nodiscard]] std::string Status() const;
private:
    struct Impl;
    std::unique_ptr<Impl> Impl_;
};
}
