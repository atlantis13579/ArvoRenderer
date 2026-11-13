#include "Distributed/FilmTransfer.h"
#include "Distributed/Protocol.h"
#include <limits>

namespace Arvo::Distributed
{
namespace
{
constexpr uint32_t FilmMagic = 0x32465241; // Little-endian "ARF2".
constexpr uint32_t HasGeometry = 1, PinholeGeometry = 2;
constexpr uint32_t MaximumFilmSide = 16384;
constexpr uint64_t MaximumFilmPixels = 64 * 1024 * 1024;
constexpr float UnitLengthTolerance = 0.002f;
bool Finite(RayTracing::Float3 v)
{
    return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z);
}
bool Unit(RayTracing::Float3 v)
{
    return Finite(v) && std::abs(RayTracing::Dot(v, v) - 1) <= UnitLengthTolerance;
}
bool KnownBackend(RayTracing::BackendType type)
{
    using B = RayTracing::BackendType;
    return type == B::CPU || type == B::CUDA || type == B::ComputeShader || type == B::DXR ||
           type == B::Metal;
}
bool MatchesBackend(const std::string& key, RayTracing::BackendType type)
{
    using B = RayTracing::BackendType;
    return key == "auto" || (key == "cpu" && type == B::CPU) || (key == "cuda" && type == B::CUDA) ||
           (key == "compute" && type == B::ComputeShader) || (key == "dxr" && type == B::DXR) ||
           (key == "metal" && type == B::Metal);
}
void Require(bool valid, const char* message)
{
    if (!valid)
        throw std::runtime_error(message);
}
void Validate(const FilmTile& tile)
{
    const auto& film = tile.Film;
    Require(film.Width && film.Height && film.Width <= MaximumFilmTileSide &&
                film.Height <= MaximumFilmTileSide && tile.FullWidth && tile.FullHeight &&
                tile.FullWidth <= MaximumFilmSide && tile.FullHeight <= MaximumFilmSide &&
                uint64_t(tile.FullWidth) * tile.FullHeight <= MaximumFilmPixels &&
                uint64_t(tile.X) + film.Width <= tile.FullWidth &&
                uint64_t(tile.Y) + film.Height <= tile.FullHeight &&
                film.Pixels.size() == size_t(film.Width) * film.Height,
            "Invalid film tile extent");
    Require(KnownBackend(tile.Backend) && RayTracing::FindIntegrator(tile.Integrator) &&
                RayTracing::FindSampler(tile.Sampling),
            "Unknown film tile transport metadata");
    std::string radiometryError;
    Require(RayTracing::ValidateRadiometry(tile.Integrator, tile.Radiometry, radiometryError),
            "Invalid film tile radiometry");
    for (const auto& pixel : film.Pixels)
        Require(Finite(pixel.MeanRadiance) &&
                    (tile.Radiometry == RayTracing::RadiometryMode::Spectral ||
                     (pixel.MeanRadiance.X >= 0 && pixel.MeanRadiance.Y >= 0 && pixel.MeanRadiance.Z >= 0)) &&
                    std::isfinite(pixel.LuminanceM2) && pixel.LuminanceM2 >= 0 &&
                    pixel.SampleCount,
                "Invalid raw film observation");
    const auto& geometry = tile.Geometry;
    if (geometry.Pixels.empty())
    {
        Require(!geometry.Width && !geometry.Height && !geometry.FullWidth && !geometry.FullHeight &&
                    !geometry.TileX && !geometry.TileY && !geometry.PinholeCorrespondence,
                "Empty guide payload has nonempty metadata");
        return;
    }
    Require(geometry.Width == film.Width && geometry.Height == film.Height &&
                geometry.FullWidth == tile.FullWidth && geometry.FullHeight == tile.FullHeight &&
                geometry.TileX == tile.X && geometry.TileY == tile.Y &&
                geometry.Pixels.size() == film.Pixels.size(),
            "Guide and film tile extents differ");
    using G = RayTracing::GeometryGuide;
    for (const auto& guide : geometry.Pixels)
    {
        Require(Unit(guide.Direction) && Finite(guide.Position) && Finite(guide.GeometricNormal) &&
                    std::isfinite(guide.CameraDepth) && std::isfinite(guide.RayDistance) &&
                    !guide.Reserved[0] && !guide.Reserved[1],
                "Invalid geometry guide values");
        if (guide.Flags == G::Background)
            Require(guide.PrimitiveIndex == RayTracing::InvalidRayPrimitiveIndex && !guide.MaterialIndex &&
                        guide.CameraDepth == 0 && guide.RayDistance == 0 &&
                        RayTracing::Dot(guide.Position, guide.Position) == 0 &&
                        RayTracing::Dot(guide.GeometricNormal, guide.GeometricNormal) == 0,
                    "Background guide contains a fabricated surface");
        else
            Require((guide.Flags == G::Surface || guide.Flags == (G::Surface | G::FrontFace)) &&
                        guide.PrimitiveIndex != RayTracing::InvalidRayPrimitiveIndex &&
                        guide.CameraDepth > 0 && guide.RayDistance > 0 && Unit(guide.GeometricNormal),
                    "Invalid surface guide identity");
    }
}
void Scalar(Http::Bytes& bytes, float value)
{
    static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559);
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    Protocol::Append32(bytes, bits);
}
void Vector(Http::Bytes& bytes, RayTracing::Float3 value)
{
    Scalar(bytes, value.X);
    Scalar(bytes, value.Y);
    Scalar(bytes, value.Z);
}
class Reader
{
public:
    explicit Reader(const Http::Bytes& bytes) : Bytes_(bytes) {}
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
    template <class T> T Enum()
    {
        const uint32_t bits = U32();
        const T value = T(bits);
        Require(uint32_t(value) == bits, "Film enum exceeds its native representation");
        return value;
    }
    float F32()
    {
        const auto bits = U32();
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    RayTracing::Float3 V3()
    {
        const float x = F32(), y = F32(), z = F32();
        return {x, y, z};
    }

private:
    const Http::Bytes& Bytes_;
    size_t At_ = 0;
};
} // namespace
const char* RenderOutputKey(RenderOutput output)
{
    switch (output)
    {
    case RenderOutput::Png:
        return "png";
    case RenderOutput::LinearFilm:
        return "film";
    case RenderOutput::GuidedFilm:
        return "guided-film";
    }
    return nullptr;
}
bool ParseRenderOutput(const std::string& key, RenderOutput& output)
{
    for (auto value : {RenderOutput::Png, RenderOutput::LinearFilm, RenderOutput::GuidedFilm})
        if (key == RenderOutputKey(value))
        {
            output = value;
            return true;
        }
    return false;
}
uint64_t FilmRequestHash(const std::string& request)
{
    constexpr uint64_t FnvOffset = 14695981039346656037ULL, FnvPrime = 1099511628211ULL;
    uint64_t hash = FnvOffset;
    for (unsigned char byte : request)
    {
        hash ^= byte;
        hash *= FnvPrime;
    }
    return hash;
}
Http::Bytes EncodeFilmTile(const FilmTile& tile)
{
    Validate(tile);
    const bool guides = !tile.Geometry.Pixels.empty();
    Http::Bytes bytes;
    bytes.reserve(FilmHeaderBytes +
                  tile.Film.Pixels.size() * (FilmPixelBytes + (guides ? GuidePixelBytes : 0)));
    Protocol::Append32(bytes, FilmMagic);
    for (uint32_t value :
         {tile.Film.Width, tile.Film.Height, tile.FullWidth, tile.FullHeight, tile.X, tile.Y,
          uint32_t(tile.Backend), uint32_t(tile.Integrator), uint32_t(tile.Radiometry),
          uint32_t(tile.Sampling), tile.SamplingSeed,
          (guides ? HasGeometry : 0) | (tile.Geometry.PinholeCorrespondence ? PinholeGeometry : 0)})
        Protocol::Append32(bytes, value);
    Protocol::Append64(bytes, tile.RenderMilliseconds);
    Protocol::Append64(bytes, tile.SceneHash);
    Protocol::Append64(bytes, tile.RequestHash);
    for (const auto& pixel : tile.Film.Pixels)
    {
        Vector(bytes, pixel.MeanRadiance);
        Scalar(bytes, pixel.LuminanceM2);
        Protocol::Append32(bytes, pixel.SampleCount);
    }
    for (const auto& guide : tile.Geometry.Pixels)
    {
        Vector(bytes, guide.Position);
        Scalar(bytes, guide.CameraDepth);
        Vector(bytes, guide.GeometricNormal);
        Scalar(bytes, guide.RayDistance);
        Vector(bytes, guide.Direction);
        Protocol::Append32(bytes, guide.Flags);
        Protocol::Append32(bytes, guide.PrimitiveIndex);
        Protocol::Append32(bytes, guide.MaterialIndex);
    }
    return bytes;
}
FilmTile DecodeFilmTile(const Http::Bytes& bytes)
{
    Require(bytes.size() >= FilmHeaderBytes && bytes.size() <= MaximumFilmTileBytes,
            "Invalid film tile packet size");
    Reader reader(bytes);
    Require(reader.U32() == FilmMagic, "Unknown film tile encoding");
    FilmTile tile;
    tile.Film.Width = reader.U32();
    tile.Film.Height = reader.U32();
    tile.FullWidth = reader.U32();
    tile.FullHeight = reader.U32();
    tile.X = reader.U32();
    tile.Y = reader.U32();
    tile.Backend = reader.Enum<RayTracing::BackendType>();
    tile.Integrator = reader.Enum<RayTracing::Algorithm>();
    tile.Radiometry = reader.Enum<RayTracing::RadiometryMode>();
    tile.Sampling = reader.Enum<RayTracing::SampleSequence>();
    tile.SamplingSeed = reader.U32();
    const uint32_t flags = reader.U32();
    Require(!(flags & ~(HasGeometry | PinholeGeometry)) &&
                (!(flags & PinholeGeometry) || (flags & HasGeometry)),
            "Unknown film tile flags");
    tile.RenderMilliseconds = reader.U64();
    tile.SceneHash = reader.U64();
    tile.RequestHash = reader.U64();
    Require(tile.Film.Width && tile.Film.Height && tile.Film.Width <= MaximumFilmTileSide &&
                tile.Film.Height <= MaximumFilmTileSide,
            "Invalid film tile packet extent");
    const size_t count = size_t(tile.Film.Width) * tile.Film.Height;
    Require(bytes.size() ==
                FilmHeaderBytes + count * (FilmPixelBytes + ((flags & HasGeometry) ? GuidePixelBytes : 0)),
            "Film tile payload length disagrees with its header");
    tile.Film.Pixels.resize(count);
    for (auto& pixel : tile.Film.Pixels)
    {
        pixel.MeanRadiance = reader.V3();
        pixel.LuminanceM2 = reader.F32();
        pixel.SampleCount = reader.U32();
    }
    if (flags & HasGeometry)
    {
        tile.Geometry = {tile.Film.Width,
                         tile.Film.Height,
                         tile.FullWidth,
                         tile.FullHeight,
                         tile.X,
                         tile.Y,
                         bool(flags & PinholeGeometry),
                         std::vector<RayTracing::GeometryGuide>(count)};
        for (auto& guide : tile.Geometry.Pixels)
        {
            guide.Position = reader.V3();
            guide.CameraDepth = reader.F32();
            guide.GeometricNormal = reader.V3();
            guide.RayDistance = reader.F32();
            guide.Direction = reader.V3();
            guide.Flags = reader.U32();
            guide.PrimitiveIndex = reader.U32();
            guide.MaterialIndex = reader.U32();
        }
    }
    Validate(tile);
    return tile;
}
FilmAssembly::FilmAssembly(const RenderConfig& config) : Config_(config)
{
    Require(config.Output == RenderOutput::LinearFilm || config.Output == RenderOutput::GuidedFilm,
            "Film assembly requires a raw output mode");
    // Spatial pilot reconstruction may allocate samples, but wire records still
    // contain only the untouched production film and its original moments.
    Require((config.Reconstruction.Filter == RayTracing::ReconstructionFilter::None &&
             !config.Adaptive.DenoiserAware) ||
                (config.Reconstruction.Filter == RayTracing::ReconstructionFilter::SpatialAtrous &&
                 config.Adaptive.Enabled && config.Adaptive.DenoiserAware),
            "Film assembly requires reconstruction after the frame barrier");
    Require(config.Width && config.Height && config.Width <= MaximumFilmSide &&
                config.Height <= MaximumFilmSide &&
                uint64_t(config.Width) * config.Height <= MaximumFilmPixels,
            "Invalid assembled film extent");
    Data_ = std::make_shared<FrameData>();
    Data_->Film = RayTracing::LinearFilm(config.Width, config.Height);
    Data_->View = config.Camera;
    Data_->Integrator = config.Integrator;
    Data_->Radiometry = config.Radiometry;
    Data_->Sampling = config.Sampling;
    Data_->SamplingSeed = config.SamplingSeed;
    if (config.Output == RenderOutput::GuidedFilm)
        Data_->Geometry = {config.Width,
                           config.Height,
                           config.Width,
                           config.Height,
                           0,
                           0,
                           config.Camera.Lens.Radius == 0 && !config.Camera.Motion.Enabled && !HasInstanceMotion(config),
                           std::vector<RayTracing::GeometryGuide>(Data_->Film.Pixels.size())};
    constexpr size_t NativeBackendCount = 5;
    Data_->Backends.reserve(NativeBackendCount);
    Covered_.resize(Data_->Film.Pixels.size());
}
void FilmAssembly::Add(const FilmTile& tile, uint64_t requestHash)
{
    Require(bool(Data_), "Film assembly has already been published");
    Validate(tile);
    Require(tile.RequestHash == requestHash && tile.FullWidth == Config_.Width &&
                tile.FullHeight == Config_.Height && tile.Integrator == Config_.Integrator &&
                tile.Radiometry == Config_.Radiometry &&
                tile.Sampling == Config_.Sampling && tile.SamplingSeed == Config_.SamplingSeed &&
                MatchesBackend(Config_.Backend, tile.Backend),
            "Worker film belongs to another render request");
    const bool guides = Config_.Output == RenderOutput::GuidedFilm;
    Require(guides == !tile.Geometry.Pixels.empty() &&
                (!guides || tile.Geometry.PinholeCorrespondence == Data_->Geometry.PinholeCorrespondence),
            "Worker geometry payload differs from the requested output");
    Require((!Config_.ExpectedSceneHash || Config_.ExpectedSceneHash == tile.SceneHash) &&
                (!HasSceneHash_ || Data_->SceneHash == tile.SceneHash),
            "Worker film scene assets differ");
    for (const auto& pixel : tile.Film.Pixels)
        Require(pixel.SampleCount <= Config_.SamplesPerPixel &&
                    (Config_.Adaptive.Enabled ? pixel.SampleCount >= Config_.Adaptive.MinimumSamples
                                              : pixel.SampleCount == Config_.SamplesPerPixel),
                "Worker film sample count differs from the requested estimator");
    for (uint32_t y = 0; y < tile.Film.Height; ++y)
        for (uint32_t x = 0; x < tile.Film.Width; ++x)
            Require(!Covered_[size_t(tile.Y + y) * Config_.Width + tile.X + x],
                    "Worker film tile overlaps an accepted tile");
    for (uint32_t y = 0; y < tile.Film.Height; ++y)
        for (uint32_t x = 0; x < tile.Film.Width; ++x)
        {
            const size_t source = size_t(y) * tile.Film.Width + x;
            const size_t target = size_t(tile.Y + y) * Config_.Width + tile.X + x;
            Data_->Film.Pixels[target] = tile.Film.Pixels[source];
            if (guides)
                Data_->Geometry.Pixels[target] = tile.Geometry.Pixels[source];
            Covered_[target] = 1;
        }
    CompletedPixels_ += tile.Film.Pixels.size();
    HasSceneHash_ = true;
    Data_->SceneHash = tile.SceneHash;
    if (std::find(Data_->Backends.begin(), Data_->Backends.end(), tile.Backend) == Data_->Backends.end())
        Data_->Backends.push_back(tile.Backend);
}
bool FilmAssembly::Complete() const
{
    return Data_ && CompletedPixels_ == Covered_.size();
}
std::shared_ptr<const FrameData> FilmAssembly::Finish()
{
    Require(Complete(), "Cannot publish an incomplete film");
    std::sort(Data_->Backends.begin(), Data_->Backends.end());
    return std::move(Data_);
}
} // namespace Arvo::Distributed
