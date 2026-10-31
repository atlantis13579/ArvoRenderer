#include "Distributed/Distributed.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace Arvo::Distributed
{
namespace
{
void CheckCancellation(const std::function<bool()>& cancelled)
{
    if (cancelled && cancelled()) throw std::runtime_error("PNG save cancelled");
}

class TemporaryFile
{
public:
    explicit TemporaryFile(const std::filesystem::path& target)
    {
        static std::atomic<std::uint64_t> sequence{0};
#ifdef _WIN32
        const auto process = GetCurrentProcessId();
#else
        const auto process = getpid();
#endif
        for (unsigned attempt = 0; attempt < 128; ++attempt)
        {
            Path_ = target;
            Path_ += ".partial-" + std::to_string(process) + "-"
                + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
                + "-" + std::to_string(sequence.fetch_add(1));
#ifdef _WIN32
            Handle_ = CreateFileW(Path_.c_str(), GENERIC_WRITE, 0, nullptr,
                CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (Handle_ != INVALID_HANDLE_VALUE) return;
            const auto error = GetLastError();
            if (error != ERROR_FILE_EXISTS && error != ERROR_ALREADY_EXISTS)
                throw std::system_error(static_cast<int>(error), std::system_category(), "Cannot create temporary PNG");
#else
            Handle_ = open(Path_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
            if (Handle_ >= 0) return;
            if (errno != EEXIST)
                throw std::system_error(errno, std::generic_category(), "Cannot create temporary PNG");
#endif
        }
        throw std::runtime_error("Cannot reserve a unique temporary PNG path");
    }

    TemporaryFile(const TemporaryFile&) = delete;
    TemporaryFile& operator=(const TemporaryFile&) = delete;

    ~TemporaryFile()
    {
        CloseHandleOnly();
        if (!Published_)
        {
            std::error_code ignored;
            std::filesystem::remove(Path_, ignored);
        }
    }

    void Write(const void* data, std::size_t length)
    {
        const auto* cursor = static_cast<const std::uint8_t*>(data);
        while (length)
        {
            const auto count = std::min<std::size_t>(length, 1024U * 1024U * 1024U);
#ifdef _WIN32
            DWORD written = 0;
            if (!WriteFile(Handle_, cursor, static_cast<DWORD>(count), &written, nullptr) || written == 0)
                throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "Writing PNG failed");
#else
            const auto written = write(Handle_, cursor, count);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0)
                throw std::system_error(errno, std::generic_category(), "Writing PNG failed");
#endif
            cursor += written;
            length -= static_cast<std::size_t>(written);
        }
    }

    void Publish(const std::filesystem::path& target, const std::function<bool()>& cancelled)
    {
        CheckCancellation(cancelled);
#ifdef _WIN32
        if (!FlushFileBuffers(Handle_))
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "Flushing PNG failed");
        CloseHandleOnly();
        for (unsigned attempt = 0; ; ++attempt)
        {
            CheckCancellation(cancelled);
            if (MoveFileExW(Path_.c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) break;
            const auto error = GetLastError();
            // Concurrent atomic replacements can briefly hold the destination entry.
            if (attempt == 39 || (error != ERROR_SHARING_VIOLATION && error != ERROR_ACCESS_DENIED))
                throw std::system_error(static_cast<int>(error), std::system_category(), "Publishing PNG failed");
            Sleep(5);
        }
#else
        if (fsync(Handle_) != 0)
            throw std::system_error(errno, std::generic_category(), "Flushing PNG failed");
        const int handle = Handle_;
        Handle_ = -1;
        if (close(handle) != 0)
            throw std::system_error(errno, std::generic_category(), "Closing PNG failed");
        CheckCancellation(cancelled);
        std::filesystem::rename(Path_, target);
#endif
        Published_ = true;
    }

private:
    void CloseHandleOnly() noexcept
    {
#ifdef _WIN32
        if (Handle_ != INVALID_HANDLE_VALUE) CloseHandle(Handle_);
        Handle_ = INVALID_HANDLE_VALUE;
#else
        if (Handle_ >= 0) close(Handle_);
        Handle_ = -1;
#endif
    }

    std::filesystem::path Path_;
    bool Published_ = false;
#ifdef _WIN32
    HANDLE Handle_ = INVALID_HANDLE_VALUE;
#else
    int Handle_ = -1;
#endif
};

void BigEndian(std::vector<std::uint8_t>& bytes, std::uint32_t value)
{
    bytes.push_back(static_cast<std::uint8_t>(value >> 24));
    bytes.push_back(static_cast<std::uint8_t>(value >> 16));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8));
    bytes.push_back(static_cast<std::uint8_t>(value));
}

std::uint32_t Crc(const std::uint8_t* bytes, std::size_t count)
{
    static const auto table = [] {
        std::array<std::uint32_t, 256> values{};
        for (std::uint32_t index = 0; index < values.size(); ++index)
        {
            auto value = index;
            for (int bit = 0; bit < 8; ++bit) value = (value >> 1) ^ ((value & 1) ? 0xedb88320U : 0U);
            values[index] = value;
        }
        return values;
    }();
    std::uint32_t value = 0xffffffffU;
    for (std::size_t index = 0; index < count; ++index) value = table[(value ^ bytes[index]) & 255U] ^ (value >> 8);
    return value ^ 0xffffffffU;
}

void Chunk(TemporaryFile& stream, const char* name, const std::vector<std::uint8_t>& data)
{
    std::vector<std::uint8_t> bytes;
    BigEndian(bytes, static_cast<std::uint32_t>(data.size()));
    bytes.insert(bytes.end(), name, name + 4);
    bytes.insert(bytes.end(), data.begin(), data.end());
    BigEndian(bytes, Crc(bytes.data() + 4, bytes.size() - 4));
    stream.Write(bytes.data(), bytes.size());
}
} // namespace

void WritePng(const std::filesystem::path& path, std::uint32_t width,
              std::uint32_t height, const std::vector<std::uint8_t>& rgba,
              const std::function<bool()>& cancelled)
{
    CheckCancellation(cancelled);
    const std::uint64_t pixelCount = std::uint64_t(width) * height;
    if (!width || !height || width > 0x7fffffffU || height > 0x7fffffffU
        || pixelCount > std::numeric_limits<std::size_t>::max() / 4
        || pixelCount * 4 != rgba.size())
        throw std::runtime_error("Invalid PNG image dimensions");
    if (path.empty()) throw std::runtime_error("PNG output path is empty");
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    TemporaryFile stream(path);
    const std::uint8_t signature[]{137, 80, 78, 71, 13, 10, 26, 10};
    stream.Write(signature, sizeof(signature));
    std::vector<std::uint8_t> header;
    BigEndian(header, width); BigEndian(header, height);
    header.insert(header.end(), {8, 6, 0, 0, 0});
    Chunk(stream, "IHDR", header);
    Chunk(stream, "sRGB", {0});
    // Stored DEFLATE blocks produce a standards-compliant PNG without a runtime codec dependency.
    const std::size_t stride = static_cast<std::size_t>(width) * 4;
    std::vector<std::uint8_t> filtered;
    filtered.reserve((stride + 1) * height);
    for (std::uint32_t y = 0; y < height; ++y)
    {
        CheckCancellation(cancelled);
        filtered.push_back(0);
        filtered.insert(filtered.end(), rgba.begin() + y * stride, rgba.begin() + (y + 1) * stride);
    }
    std::vector<std::uint8_t> zlib{0x78, 0x01};
    std::uint32_t a = 1, b = 0;
    std::size_t processed = 0;
    for (const auto value : filtered)
    {
        if ((processed++ & 65535U) == 0) CheckCancellation(cancelled);
        a = (a + value) % 65521;
        b = (b + a) % 65521;
    }
    for (std::size_t offset = 0; offset < filtered.size();)
    {
        CheckCancellation(cancelled);
        const auto count = static_cast<std::uint16_t>(std::min<std::size_t>(65535, filtered.size() - offset));
        zlib.push_back(offset + count == filtered.size() ? 1 : 0);
        zlib.push_back(static_cast<std::uint8_t>(count));
        zlib.push_back(static_cast<std::uint8_t>(count >> 8));
        const auto inverse = static_cast<std::uint16_t>(~count);
        zlib.push_back(static_cast<std::uint8_t>(inverse));
        zlib.push_back(static_cast<std::uint8_t>(inverse >> 8));
        zlib.insert(zlib.end(), filtered.begin() + offset, filtered.begin() + offset + count);
        offset += count;
    }
    BigEndian(zlib, (b << 16) | a);
    Chunk(stream, "IDAT", zlib);
    Chunk(stream, "IEND", {});
    stream.Publish(path, cancelled);
}
} // namespace Arvo::Distributed
