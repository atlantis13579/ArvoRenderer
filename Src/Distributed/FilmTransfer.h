#pragma once
#include "Distributed/Distributed.h"
#include "Distributed/Http.h"

namespace Arvo::Distributed
{
// A portable wire record, not a memcpy of native structs or GPU padding.
struct FilmTile
{
    RayTracing::LinearFilm Film;
    RayTracing::GeometryGuideImage Geometry;
    uint32_t FullWidth = 0, FullHeight = 0, X = 0, Y = 0;
    RayTracing::BackendType Backend = RayTracing::BackendType::CPU;
    RayTracing::Algorithm Integrator = RayTracing::Algorithm::PathTracing;
    RayTracing::RadiometryMode Radiometry = RayTracing::RadiometryMode::RGB;
    RayTracing::SampleSequence Sampling = RayTracing::SampleSequence::Independent;
    uint32_t SamplingSeed = 0;
    uint64_t RenderMilliseconds = 0, SceneHash = 0, RequestHash = 0;
};

inline constexpr size_t FilmHeaderBytes = 76, FilmPixelBytes = 20, GuidePixelBytes = 56;
inline constexpr uint32_t MaximumFilmTileSide = 1024;
inline constexpr size_t MaximumFilmTileBytes =
    FilmHeaderBytes + size_t(MaximumFilmTileSide) * MaximumFilmTileSide * (FilmPixelBytes + GuidePixelBytes);
uint64_t FilmRequestHash(const std::string& canonicalRequest);
Http::Bytes EncodeFilmTile(const FilmTile& tile);
FilmTile DecodeFilmTile(const Http::Bytes& bytes);

// Caller serializes Add calls. Validation and overlap checks finish before any
// destination pixels change; an invalid retry cannot leave a partial tile.
class FilmAssembly
{
public:
    explicit FilmAssembly(const RenderConfig& config);
    void Add(const FilmTile& tile, uint64_t requestHash);
    bool Complete() const;
    std::shared_ptr<const FrameData> Finish();

private:
    RenderConfig Config_;
    std::shared_ptr<FrameData> Data_;
    std::vector<uint8_t> Covered_;
    size_t CompletedPixels_ = 0;
    bool HasSceneHash_ = false;
};
} // namespace Arvo::Distributed
