#pragma once
#include "Distributed/Http.h"
#include "Renderer/TemporalReconstruction.h"
#include <optional>

namespace Arvo::Distributed
{
inline constexpr size_t MaximumTemporalPixels = size_t(64) * 1024 * 1024;
inline constexpr size_t MaximumTemporalMetadata = 64 * 1024;
// Current film/geometry: 20+56 bytes. Reused history/geometry: 16+56 bytes.
inline constexpr size_t MaximumTemporalRequestBytes =
    8 + MaximumTemporalMetadata + MaximumTemporalPixels * 148;
inline constexpr size_t MaximumTemporalResponseBytes = 68 + MaximumTemporalPixels * 20;
inline constexpr const char* TemporalWireVersion = "2";

struct TemporalPacket
{
    std::string Job;
    RayTracing::BackendType Backend = RayTracing::BackendType::CPU;
    RayTracing::LinearFilm Film;
    RayTracing::GeometryGuideImage Geometry;
    RayTracing::Camera View;
    RayTracing::TemporalIdentity Identity;
    RayTracing::TemporalOptions Options;
    std::optional<RayTracing::TemporalHistory> Previous;
    RayTracing::TemporalReset Reset = RayTracing::TemporalReset::NoHistory;
    uint64_t RequestHash = 0;
    RayTracing::TemporalFrame Frame() const { return {Film, Geometry, View, Identity}; }
};

uint64_t TemporalPacketHash(const Http::Bytes&, const RayTracing::CancellationToken& = {});
Http::Bytes EncodeTemporalRequest(const std::string& job, RayTracing::BackendType,
                                  const RayTracing::TemporalFrame&, const RayTracing::TemporalHistory*,
                                  const RayTracing::TemporalOptions&,
                                  const RayTracing::CancellationToken& = {});
TemporalPacket DecodeTemporalRequest(const Http::Bytes&, const RayTracing::CancellationToken& = {});
Http::Bytes EncodeTemporalResponse(const TemporalPacket&, const RayTracing::TemporalResult&,
                                   const RayTracing::CancellationToken& = {});
RayTracing::TemporalResult DecodeTemporalResponse(const Http::Bytes&, uint64_t requestHash,
                                                  RayTracing::BackendType, const RayTracing::TemporalFrame&,
                                                  const RayTracing::TemporalOptions&,
                                                  RayTracing::TemporalReset,
                                                  const RayTracing::CancellationToken& = {});

struct RemoteTemporalConfig
{
    std::string Server, Token;
    RayTracing::BackendType Backend = RayTracing::BackendType::CPU;
    int TimeoutSeconds = 600;
};
// Transport only: the worker invokes the selected native implementation. History
// is supplied by the caller and returned as a candidate, never retained remotely.
RayTracing::TemporalResult ReconstructRemoteTemporal(const RemoteTemporalConfig&,
                                                     const RayTracing::TemporalFrame&,
                                                     const RayTracing::TemporalHistory*,
                                                     const RayTracing::TemporalOptions&,
                                                     const RayTracing::CancellationToken& = {});
} // namespace Arvo::Distributed
