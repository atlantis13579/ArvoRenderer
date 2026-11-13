#include "Distributed/TemporalTransfer.h"
#include "Distributed/Protocol.h"
#include <chrono>
#include <random>

namespace Arvo::Distributed
{
namespace
{
namespace R = RayTracing;
using Fields = std::map<std::string, std::string>;
constexpr uint32_t RequestMagic = 0x31545241, ResponseMagic = 0x31535241; // ART1 / ARS1.
constexpr size_t PrefixBytes = 8, ResponseHeaderBytes = 68;
constexpr size_t FilmStride = 20, GuideStride = 56, HistoryStride = 16, DisplayStride = 4;
constexpr uint32_t MaximumSide = 16384;
constexpr size_t CancellationStride = 256;
void Require(bool valid, const std::string& error)
{
    if (!valid)
        throw std::runtime_error(error);
}
void CheckCancellation(const R::CancellationToken& token, size_t index = 0)
{
    if (index % CancellationStride == 0 && token.IsCancellationRequested())
        throw std::runtime_error("Temporal transfer cancelled");
}
size_t PixelCount(uint32_t width, uint32_t height)
{
    const uint64_t count = uint64_t(width) * height;
    Require(width && height && width <= MaximumSide && height <= MaximumSide &&
                count <= MaximumTemporalPixels,
            "Temporal transfer extent exceeds the full-film contract");
    return size_t(count);
}
bool KnownBackend(R::BackendType backend)
{
    using B = R::BackendType;
    return backend == B::CPU || backend == B::CUDA || backend == B::ComputeShader || backend == B::DXR ||
           backend == B::Metal;
}
uint64_t U64(const Fields& fields, const std::string& key)
{
    const auto& text = fields.at(key);
    Require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos,
            "Invalid temporal integer: " + key);
    return std::stoull(text);
}
uint32_t U32(const Fields& fields, const std::string& key, uint32_t maximum = UINT32_MAX)
{
    return Protocol::UInt(fields, key, 0, maximum);
}
R::BackendType Backend(const Fields& fields, const std::string& key)
{
    const auto bits = U32(fields, key, UINT8_MAX);
    const auto backend = R::BackendType(bits);
    Require(KnownBackend(backend), "Unknown native temporal backend");
    return backend;
}
float MetadataFloat(const Fields& fields, const std::string& key)
{
    // Parsing through double preserves float subnormals without std::stof's
    // underflow exception, and the classic locale matches Protocol::Number.
    std::istringstream stream(fields.at(key));
    stream.imbue(std::locale::classic());
    double value = 0;
    stream >> value;
    Require(bool(stream) && stream.peek() == std::char_traits<char>::eof() && std::isfinite(value) &&
                std::abs(value) <= std::numeric_limits<float>::max(),
            "Invalid temporal scalar: " + key);
    return float(value);
}
void PutIdentity(Fields& fields, const std::string& prefix, const R::TemporalIdentity& id)
{
    fields[prefix + "sequence"] = std::to_string(id.Sequence);
    fields[prefix + "content"] = std::to_string(id.Content);
    fields[prefix + "frame"] = std::to_string(id.Frame);
    fields[prefix + "samples"] = std::to_string(id.Samples);
}
R::TemporalIdentity GetIdentity(const Fields& fields, const std::string& prefix)
{
    return {U64(fields, prefix + "sequence"), U64(fields, prefix + "content"), U64(fields, prefix + "frame"),
            U64(fields, prefix + "samples")};
}
void PutCamera(Fields& fields, const std::string& prefix, const R::Camera& camera)
{
    fields[prefix + "exposure"] = Protocol::EncodeCameraExposure(camera.Motion);
    const float values[] = {camera.Position.X,  camera.Position.Y,
                            camera.Position.Z,  camera.Target.X,
                            camera.Target.Y,    camera.Target.Z,
                            camera.Up.X,        camera.Up.Y,
                            camera.Up.Z,        camera.VerticalFieldOfViewDegrees,
                            camera.Lens.Radius, camera.Lens.FocusDistance};
    constexpr const char* names[] = {"px", "py", "pz", "tx",  "ty",   "tz",
                                     "ux", "uy", "uz", "fov", "lens", "focus"};
    for (size_t i = 0; i < std::size(values); ++i)
        fields[prefix + names[i]] = Protocol::Number(values[i]);
}
R::Camera GetCamera(const Fields& fields, const std::string& prefix)
{
    const auto value = [&](const char* name) { return MetadataFloat(fields, prefix + name); };
    R::Camera camera;
    camera.Position = {value("px"), value("py"), value("pz")};
    camera.Target = {value("tx"), value("ty"), value("tz")};
    camera.Up = {value("ux"), value("uy"), value("uz")};
    camera.VerticalFieldOfViewDegrees = value("fov");
    camera.Lens.Radius = value("lens");
    camera.Lens.FocusDistance = value("focus");
    camera.Motion = Protocol::ParseCameraExposure(fields.at(prefix + "exposure"));
    return camera;
}
void PutOptions(Fields& fields, const std::string& prefix, const R::TemporalOptions& options)
{
    fields[prefix + "history_cap"] = std::to_string(options.MaximumHistory);
    fields[prefix + "gamma"] = options.GammaCorrection ? "1" : "0";
    const float values[] = {options.MinimumCurrentWeight,   options.AbsoluteDepthTolerance,
                            options.RelativeDepthTolerance, options.MinimumNormalCosine,
                            options.MinimumHistoryWeight,   options.NeighborhoodExpansion};
    constexpr const char* names[] = {"fresh", "depth_abs", "depth_rel", "normal", "support", "clip"};
    for (size_t i = 0; i < std::size(values); ++i)
        fields[prefix + names[i]] = Protocol::Number(values[i]);
}
R::TemporalOptions GetOptions(const Fields& fields, const std::string& prefix)
{
    R::TemporalOptions options;
    options.MaximumHistory = U32(fields, prefix + "history_cap");
    options.GammaCorrection = U32(fields, prefix + "gamma", 1) != 0;
    options.MinimumCurrentWeight = MetadataFloat(fields, prefix + "fresh");
    options.AbsoluteDepthTolerance = MetadataFloat(fields, prefix + "depth_abs");
    options.RelativeDepthTolerance = MetadataFloat(fields, prefix + "depth_rel");
    options.MinimumNormalCosine = MetadataFloat(fields, prefix + "normal");
    options.MinimumHistoryWeight = MetadataFloat(fields, prefix + "support");
    options.NeighborhoodExpansion = MetadataFloat(fields, prefix + "clip");
    return options;
}
void Scalar(Http::Bytes& bytes, float value)
{
    static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559);
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    Protocol::Append32(bytes, bits);
}
void Vector(Http::Bytes& bytes, R::Float3 value)
{
    Scalar(bytes, value.X);
    Scalar(bytes, value.Y);
    Scalar(bytes, value.Z);
}
void Guides(Http::Bytes& bytes, const std::vector<R::GeometryGuide>& guides,
            const R::CancellationToken& token)
{
    for (size_t i = 0; i < guides.size(); ++i)
    {
        CheckCancellation(token, i);
        const auto& guide = guides[i];
        Vector(bytes, guide.Position);
        Scalar(bytes, guide.CameraDepth);
        Vector(bytes, guide.GeometricNormal);
        Scalar(bytes, guide.RayDistance);
        Vector(bytes, guide.Direction);
        Protocol::Append32(bytes, guide.Flags);
        Protocol::Append32(bytes, guide.PrimitiveIndex);
        Protocol::Append32(bytes, guide.MaterialIndex);
    }
}
void History(Http::Bytes& bytes, const std::vector<R::TemporalPixel>& pixels,
             const R::CancellationToken& token)
{
    for (size_t i = 0; i < pixels.size(); ++i)
    {
        CheckCancellation(token, i);
        Vector(bytes, pixels[i].Radiance);
        Protocol::Append32(bytes, pixels[i].Age);
    }
}
class Reader
{
public:
    explicit Reader(const Http::Bytes& bytes, size_t start = 0) : Bytes_(bytes), At_(start) {}
    uint32_t U32()
    {
        const auto value = Protocol::Read32(Bytes_, At_);
        At_ += sizeof(value);
        return value;
    }
    uint64_t U64()
    {
        const auto value = Protocol::Read64(Bytes_, At_);
        At_ += sizeof(value);
        return value;
    }
    float F32()
    {
        const auto bits = U32();
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    R::Float3 V3()
    {
        const float x = F32(), y = F32(), z = F32();
        return {x, y, z};
    }
    void Guides(std::vector<R::GeometryGuide>& guides, const R::CancellationToken& token)
    {
        for (size_t i = 0; i < guides.size(); ++i)
        {
            CheckCancellation(token, i);
            auto& guide = guides[i];
            guide.Position = V3();
            guide.CameraDepth = F32();
            guide.GeometricNormal = V3();
            guide.RayDistance = F32();
            guide.Direction = V3();
            guide.Flags = U32();
            guide.PrimitiveIndex = U32();
            guide.MaterialIndex = U32();
        }
    }
    void History(std::vector<R::TemporalPixel>& pixels, const R::CancellationToken& token)
    {
        for (size_t i = 0; i < pixels.size(); ++i)
        {
            CheckCancellation(token, i);
            pixels[i].Radiance = V3();
            pixels[i].Age = U32();
        }
    }

private:
    const Http::Bytes& Bytes_;
    size_t At_;
};
bool SameIdentity(const R::TemporalIdentity& a, const R::TemporalIdentity& b)
{
    return a.Sequence == b.Sequence && a.Content == b.Content && a.Frame == b.Frame && a.Samples == b.Samples;
}
uint64_t ValidatePixels(const std::vector<R::TemporalPixel>& pixels, const R::TemporalOptions& options)
{
    uint64_t accepted = 0;
    for (const auto& pixel : pixels)
    {
        const auto p = pixel.Radiance;
        Require(std::isfinite(p.X) && std::isfinite(p.Y) && std::isfinite(p.Z) &&
                    pixel.Age && pixel.Age <= options.MaximumHistory,
                "Invalid temporal readback radiance or age");
        accepted += pixel.Age > 1;
    }
    return accepted;
}
void ValidateResult(const R::TemporalResult& result, R::BackendType backend, const R::TemporalFrame& frame,
                    const R::TemporalOptions& options, R::TemporalReset reset)
{
    Require(result.Status == R::RenderStatus::Completed && result.Reset == reset &&
                result.History.Backend == backend && SameIdentity(result.History.Identity, frame.Identity) &&
                result.History.Options == options &&
                result.History.Pixels.size() == frame.Film.Pixels.size() &&
                result.Display.GetWidth() == frame.Film.Width &&
                result.Display.GetHeight() == frame.Film.Height &&
                result.Display.GetPixelsRGBA8().size() == frame.Film.Pixels.size() * DisplayStride,
            "Temporal readback belongs to another frame or native backend");
    Require(result.AcceptedHistoryPixels == ValidatePixels(result.History.Pixels, options) &&
                (reset == R::TemporalReset::None || !result.AcceptedHistoryPixels),
            "Temporal readback misreported accepted history");
}
} // namespace
uint64_t TemporalPacketHash(const Http::Bytes& bytes, const R::CancellationToken& token)
{
    constexpr uint64_t FnvOffset = 14695981039346656037ULL, FnvPrime = 1099511628211ULL;
    constexpr size_t HashBlockBytes = 64 * 1024;
    uint64_t hash = FnvOffset;
    for (size_t begin = 0; begin < bytes.size(); begin += HashBlockBytes)
    {
        CheckCancellation(token);
        const size_t end = std::min(bytes.size(), begin + HashBlockBytes);
        for (size_t i = begin; i < end; ++i)
        {
            hash ^= bytes[i];
            hash *= FnvPrime;
        }
    }
    return hash;
}
Http::Bytes EncodeTemporalRequest(const std::string& job, R::BackendType backend,
                                  const R::TemporalFrame& frame, const R::TemporalHistory* previous,
                                  const R::TemporalOptions& options, const R::CancellationToken& token)
{
    CheckCancellation(token);
    Require(!job.empty() && job.size() <= 128 && job.find_first_of("\r\n") == std::string::npos,
            "Invalid temporal job identifier");
    Require(KnownBackend(backend), "Unknown native temporal backend");
    R::TemporalReset reset;
    std::string error;
    Require(R::Detail::ValidateTemporalRequest(backend, frame, previous, options, reset, error), error);
    const size_t count = PixelCount(frame.Film.Width, frame.Film.Height);
    const bool reuse = reset == R::TemporalReset::None;
    Fields fields{{"version", Protocol::Version},
                  {"wire", TemporalWireVersion},
                  {"job", job},
                  {"backend", std::to_string(uint32_t(backend))},
                  {"width", std::to_string(frame.Film.Width)},
                  {"height", std::to_string(frame.Film.Height)},
                  {"history", previous ? "1" : "0"},
                  {"reuse", reuse ? "1" : "0"}};
    PutIdentity(fields, "", frame.Identity);
    PutCamera(fields, "", frame.View);
    PutOptions(fields, "", options);
    if (previous)
    {
        fields["old_backend"] = std::to_string(uint32_t(previous->Backend));
        fields["old_width"] = std::to_string(previous->Geometry.Width);
        fields["old_height"] = std::to_string(previous->Geometry.Height);
        PutIdentity(fields, "old_", previous->Identity);
        PutCamera(fields, "old_", previous->View);
        PutOptions(fields, "old_", previous->Options);
    }
    const std::string metadata = Http::EncodeFields(fields);
    Require(metadata.size() <= MaximumTemporalMetadata, "Temporal metadata exceeds limit");
    Http::Bytes bytes;
    bytes.reserve(PrefixBytes + metadata.size() +
                  count * (FilmStride + GuideStride + (reuse ? HistoryStride + GuideStride : 0)));
    Protocol::Append32(bytes, RequestMagic);
    Protocol::Append32(bytes, uint32_t(metadata.size()));
    bytes.insert(bytes.end(), metadata.begin(), metadata.end());
    for (size_t i = 0; i < count; ++i)
    {
        CheckCancellation(token, i);
        Vector(bytes, frame.Film.Pixels[i].MeanRadiance);
        Scalar(bytes, frame.Film.Pixels[i].LuminanceM2);
        Protocol::Append32(bytes, frame.Film.Pixels[i].SampleCount);
    }
    Guides(bytes, frame.Geometry.Pixels, token);
    if (reuse)
    {
        History(bytes, previous->Pixels, token);
        Guides(bytes, previous->Geometry.Pixels, token);
    }
    return bytes;
}
TemporalPacket DecodeTemporalRequest(const Http::Bytes& bytes, const R::CancellationToken& token)
{
    CheckCancellation(token);
    Require(bytes.size() >= PrefixBytes && bytes.size() <= MaximumTemporalRequestBytes,
            "Invalid temporal request size");
    Reader prefix(bytes);
    Require(prefix.U32() == RequestMagic, "Unknown temporal request encoding");
    const size_t metadataBytes = prefix.U32();
    Require(metadataBytes && metadataBytes <= MaximumTemporalMetadata &&
                PrefixBytes + metadataBytes <= bytes.size(),
            "Truncated temporal metadata");
    const auto fields = Http::ParseFields(
        std::string(bytes.begin() + PrefixBytes, bytes.begin() + PrefixBytes + metadataBytes));
    Require(fields.at("version") == Protocol::Version && fields.at("wire") == TemporalWireVersion,
            "Unsupported temporal transport version");
    TemporalPacket packet;
    packet.Job = fields.at("job");
    Require(!packet.Job.empty() && packet.Job.size() <= 128 &&
                packet.Job.find_first_of("\r\n") == std::string::npos,
            "Invalid temporal job identifier");
    packet.Backend = Backend(fields, "backend");
    packet.Film.Width = U32(fields, "width", MaximumSide);
    packet.Film.Height = U32(fields, "height", MaximumSide);
    const size_t count = PixelCount(packet.Film.Width, packet.Film.Height);
    packet.Identity = GetIdentity(fields, "");
    packet.View = GetCamera(fields, "");
    packet.Options = GetOptions(fields, "");
    const bool hasHistory = U32(fields, "history", 1) != 0, reuse = U32(fields, "reuse", 1) != 0;
    Require(!reuse || hasHistory, "Temporal reuse requires history metadata");
    if (hasHistory)
    {
        auto& previous = packet.Previous.emplace();
        previous.Backend = Backend(fields, "old_backend");
        previous.Identity = GetIdentity(fields, "old_");
        previous.View = GetCamera(fields, "old_");
        previous.Options = GetOptions(fields, "old_");
        previous.Geometry.Width = previous.Geometry.FullWidth = U32(fields, "old_width", MaximumSide);
        previous.Geometry.Height = previous.Geometry.FullHeight = U32(fields, "old_height", MaximumSide);
        previous.Geometry.PinholeCorrespondence = true;
        if (reuse)
            Require(PixelCount(previous.Geometry.Width, previous.Geometry.Height) == count,
                    "Reused temporal history extent differs from its observation");
    }
    Require(bytes.size() ==
                PrefixBytes + metadataBytes +
                    count * (FilmStride + GuideStride + (reuse ? HistoryStride + GuideStride : 0)),
            "Temporal payload length disagrees with its metadata");
    Reader reader(bytes, PrefixBytes + metadataBytes);
    packet.Film.Pixels.resize(count);
    for (size_t i = 0; i < count; ++i)
    {
        CheckCancellation(token, i);
        auto& pixel = packet.Film.Pixels[i];
        pixel.MeanRadiance = reader.V3();
        pixel.LuminanceM2 = reader.F32();
        pixel.SampleCount = reader.U32();
    }
    packet.Geometry = {packet.Film.Width,
                       packet.Film.Height,
                       packet.Film.Width,
                       packet.Film.Height,
                       0,
                       0,
                       true,
                       std::vector<R::GeometryGuide>(count)};
    reader.Guides(packet.Geometry.Pixels, token);
    if (reuse)
    {
        packet.Previous->Pixels.resize(count);
        packet.Previous->Geometry.Pixels.resize(count);
        reader.History(packet.Previous->Pixels, token);
        reader.Guides(packet.Previous->Geometry.Pixels, token);
    }
    std::string error;
    Require(R::Detail::ValidateTemporalRequest(packet.Backend, packet.Frame(),
                                               packet.Previous ? &*packet.Previous : nullptr, packet.Options,
                                               packet.Reset, error),
            error);
    Require(reuse == (packet.Reset == R::TemporalReset::None),
            "Temporal history payload uses a false reuse claim");
    CheckCancellation(token);
    packet.RequestHash = TemporalPacketHash(bytes, token);
    return packet;
}
Http::Bytes EncodeTemporalResponse(const TemporalPacket& packet, const R::TemporalResult& result,
                                   const R::CancellationToken& token)
{
    CheckCancellation(token);
    ValidateResult(result, packet.Backend, packet.Frame(), packet.Options, packet.Reset);
    const auto same = [](R::Float3 a, R::Float3 b) { return a.X == b.X && a.Y == b.Y && a.Z == b.Z; };
    const auto& view = result.History.View;
    const auto& geometry = result.History.Geometry;
    Require(same(view.Position, packet.View.Position) && same(view.Target, packet.View.Target) &&
                same(view.Up, packet.View.Up) &&
                view.VerticalFieldOfViewDegrees == packet.View.VerticalFieldOfViewDegrees &&
                view.Lens == packet.View.Lens && view.Motion == packet.View.Motion &&
                geometry.Width == packet.Geometry.Width && geometry.Height == packet.Geometry.Height &&
                geometry.FullWidth == geometry.Width && geometry.FullHeight == geometry.Height &&
                !geometry.TileX && !geometry.TileY && geometry.PinholeCorrespondence &&
                geometry.Pixels.size() == packet.Geometry.Pixels.size() &&
                std::memcmp(geometry.Pixels.data(), packet.Geometry.Pixels.data(),
                            geometry.Pixels.size() * sizeof(R::GeometryGuide)) == 0,
            "Temporal readback changed current geometric correspondence");
    Http::Bytes bytes;
    bytes.reserve(ResponseHeaderBytes + result.History.Pixels.size() * (HistoryStride + DisplayStride));
    Protocol::Append32(bytes, ResponseMagic);
    Protocol::Append64(bytes, packet.RequestHash);
    Protocol::Append32(bytes, packet.Film.Width);
    Protocol::Append32(bytes, packet.Film.Height);
    Protocol::Append32(bytes, uint32_t(packet.Backend));
    Protocol::Append32(bytes, uint32_t(result.Reset));
    Protocol::Append64(bytes, result.AcceptedHistoryPixels);
    for (uint64_t value : {result.History.Identity.Sequence, result.History.Identity.Content,
                           result.History.Identity.Frame, result.History.Identity.Samples})
        Protocol::Append64(bytes, value);
    History(bytes, result.History.Pixels, token);
    CheckCancellation(token);
    const auto& display = result.Display.GetPixelsRGBA8();
    bytes.insert(bytes.end(), display.begin(), display.end());
    return bytes;
}
R::TemporalResult DecodeTemporalResponse(const Http::Bytes& bytes, uint64_t requestHash,
                                         R::BackendType backend, const R::TemporalFrame& frame,
                                         const R::TemporalOptions& options, R::TemporalReset reset,
                                         const R::CancellationToken& token)
{
    CheckCancellation(token);
    const size_t count = PixelCount(frame.Film.Width, frame.Film.Height);
    Require(bytes.size() == ResponseHeaderBytes + count * (HistoryStride + DisplayStride),
            "Truncated or oversized temporal response");
    Reader reader(bytes);
    Require(reader.U32() == ResponseMagic && reader.U64() == requestHash &&
                reader.U32() == frame.Film.Width && reader.U32() == frame.Film.Height &&
                reader.U32() == uint32_t(backend) && reader.U32() == uint32_t(reset),
            "Temporal response belongs to another native request");
    R::TemporalResult result;
    result.Status = R::RenderStatus::Completed;
    result.Reset = reset;
    result.AcceptedHistoryPixels = reader.U64();
    result.History.Identity = {reader.U64(), reader.U64(), reader.U64(), reader.U64()};
    result.History.Backend = backend;
    result.History.View = frame.View;
    result.History.Options = options;
    result.History.Pixels.resize(count);
    reader.History(result.History.Pixels, token);
    result.Display = R::Image(frame.Film.Width, frame.Film.Height);
    const size_t firstDisplay = ResponseHeaderBytes + count * HistoryStride;
    std::copy(bytes.begin() + firstDisplay, bytes.end(), result.Display.GetPixelsRGBA8().begin());
    ValidateResult(result, backend, frame, options, reset);
    CheckCancellation(token);
    // Correspondence is unchanged by reconstruction; avoid sending it back again.
    result.History.Geometry = frame.Geometry;
    CheckCancellation(token);
    return result;
}
R::TemporalResult ReconstructRemoteTemporal(const RemoteTemporalConfig& config, const R::TemporalFrame& frame,
                                            const R::TemporalHistory* previous,
                                            const R::TemporalOptions& options,
                                            const R::CancellationToken& token)
{
    R::TemporalResult result;
    std::string job;
    try
    {
        CheckCancellation(token);
        Require(config.TimeoutSeconds > 0 && config.TimeoutSeconds <= 3600,
                "Invalid temporal worker timeout");
        const auto cancelled = [&] { return token.IsCancellationRequested(); };
        const auto status = Http::RequestBinary(config.Server, "GET", "/status", {}, config.Token,
                                                config.TimeoutSeconds, cancelled, MaximumTemporalMetadata);
        Require(status.Status == 200, "Temporal worker status: HTTP " + std::to_string(status.Status));
        const auto fields = Http::ParseFields(Http::Text(status.Body));
        Require(fields.at("version") == Protocol::Version && fields.count("temporal_wire") &&
                    fields.at("temporal_wire") == TemporalWireVersion,
                "Worker does not support native temporal transfer");
        std::random_device random;
        job = "temporal-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
              "-" + std::to_string(random());
        auto bytes = EncodeTemporalRequest(job, config.Backend, frame, previous, options, token);
        const uint64_t hash = TemporalPacketHash(bytes, token);
        R::TemporalReset reset;
        std::string error;
        Require(R::Detail::ValidateTemporalRequest(config.Backend, frame, previous, options, reset, error),
                error);
        const size_t responseLimit = std::max(MaximumTemporalMetadata,
            ResponseHeaderBytes + frame.Film.Pixels.size() * (HistoryStride + DisplayStride));
        const auto response =
            Http::RequestBinary(config.Server, "POST", "/temporal", bytes, config.Token,
                                config.TimeoutSeconds, cancelled, responseLimit, job);
        Http::Bytes{}.swap(bytes);
        if (response.Status != 200)
            throw std::runtime_error("Temporal worker HTTP " + std::to_string(response.Status) + ": " +
                                     Http::Text(response.Body));
        result = DecodeTemporalResponse(response.Body, hash, config.Backend, frame, options, reset, token);
        CheckCancellation(token);
    }
    catch (const std::exception& error)
    {
        result = {};
        result.Status =
            token.IsCancellationRequested() ? R::RenderStatus::Cancelled : R::RenderStatus::Failed;
        result.Error = error.what();
        if (!job.empty())
            try
            {
                Http::Request(config.Server, "POST", "/cancel", Http::EncodeFields({{"job", job}}),
                              config.Token, 2);
            }
            catch (...)
            {
            }
    }
    return result;
}
} // namespace Arvo::Distributed
