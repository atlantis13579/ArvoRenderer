#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

namespace Arvo::Core
{

inline std::filesystem::path ExecutableDirectory()
{
    std::filesystem::path executable;
#ifdef _WIN32
    for (DWORD capacity = 512; capacity <= 32768; capacity *= 2)
    {
        std::vector<wchar_t> buffer(capacity);
        const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), capacity);
        if (!length) break;
        if (length < capacity)
        {
            executable = std::wstring(buffer.data(), length);
            break;
        }
    }
#elif defined(__APPLE__)
    std::uint32_t length = 0;
    _NSGetExecutablePath(nullptr, &length);
    if (length)
    {
        std::vector<char> buffer(length);
        if (_NSGetExecutablePath(buffer.data(), &length) == 0) executable = buffer.data();
    }
#else
    for (std::size_t capacity = 512; capacity <= 32768; capacity *= 2)
    {
        std::vector<char> buffer(capacity);
        const auto length = readlink("/proc/self/exe", buffer.data(), buffer.size());
        if (length < 0) break;
        if (static_cast<std::size_t>(length) < capacity)
        {
            executable = std::string(buffer.data(), static_cast<std::size_t>(length));
            break;
        }
    }
#endif
    std::error_code error;
    if (!executable.empty())
    {
        const auto canonical = std::filesystem::weakly_canonical(executable, error);
        return (error ? executable : canonical).parent_path();
    }
    const auto current = std::filesystem::current_path(error);
    return error ? std::filesystem::path(".") : current;
}

inline bool HasScenes(const std::filesystem::path& directory)
{
    std::error_code error;
    return std::filesystem::is_directory(directory / "Scenes" / "CornellBox", error) &&
           std::filesystem::is_directory(directory / "Scenes" / "Sponza", error);
}

inline std::filesystem::path ContentRoot()
{
    // Support a flat deployment and the repository's Binaries/Release layout.
    auto directory = ExecutableDirectory();
    for (unsigned level = 0; level < 3 && !directory.empty(); ++level)
    {
        const auto candidate = directory / "Contents";
        if (HasScenes(candidate)) return candidate;
        directory = directory.parent_path();
    }
#ifdef ARVO_CONTENT_ROOT
    return std::filesystem::path(ARVO_CONTENT_ROOT);
#else
    return ExecutableDirectory() / "Contents";
#endif
}

inline std::filesystem::path BinaryRoot()
{
    const auto binaries = ContentRoot().parent_path() / "Binaries";
    std::error_code error;
    if (std::filesystem::is_directory(binaries, error)) return binaries;
    return ExecutableDirectory();
}

} // namespace Arvo::Core
